import os
import json
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms as T
from torchvision.models.detection import keypointrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BASE           = "/data01/giacomelli/data"
ANN            = f"{BASE}/RHPE_Annotations"
KPT_CHECKPOINT_DIR = "/data01/giacomelli/experiments/keypoints/checkpoints"
DET_CHECKPOINT      = "/data01/giacomelli/experiments/bbox_detection/checkpoints/fasterrcnn_rhpe.pth"

NUM_KEYPOINTS = 17
CATEGORY_ID   = 1
# Uniform OKS sigma for all 17 keypoints.
# COCO's official sigmas (calibrated on human body keypoints) range
# roughly from 0.025 (eyes) to 0.107 (hips) — no equivalent values
# exist for hand anatomical keypoints, so we use a single mid-range
# value as a documented simplification.
KPT_OKS_SIGMA = 0.05

SPLIT_CONFIG = {
    "train": {
        "image_dir": f"{BASE}/RHPE_train",
        "json_path": f"{ANN}/RHPE_anatomical_ROIs_train.json",
    },
    "val": {
        "image_dir": f"{BASE}/RHPE_val",
        "json_path": f"{ANN}/RHPE_anatomical_ROIs_val.json",
    },
    "test": {
        "image_dir": f"{BASE}/RHPE_test",
        "json_path": f"{ANN}/RHPE_anatomical_ROIs_test.json",
    },
}


# ── Dataset ──────────────────────────────────────────────────────────────────
class RHPEKeypointEvalDataset(Dataset):
    """Dataset for inference only: returns (PIL image, coco_image_id)."""

    def __init__(self, image_dir, json_path):
        self.image_dir = image_dir

        with open(json_path) as f:
            coco = json.load(f)

        self.id_to_file = {img['id']: img['file_name'] for img in coco['images']}
        self.id_to_ann  = {}
        for ann in coco['annotations']:
            if ann['image_id'] not in self.id_to_ann:
                self.id_to_ann[ann['image_id']] = ann

        self.image_ids = list(self.id_to_file.keys())
        print(f"Dataset: {len(self.image_ids)} images")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        coco_id = self.image_ids[idx]
        fname   = self.id_to_file[coco_id]
        img_pil = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
        return img_pil, coco_id


# ── Models ───────────────────────────────────────────────────────────────────
def get_kpt_model(num_keypoints=NUM_KEYPOINTS):
    model       = keypointrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_features, num_keypoints)
    return model


def get_det_model(num_classes=2):
    model       = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'])
    parser.add_argument('--model', type=str, required=True, choices=['full', 'gt_crop'],
                        help='which trained checkpoint to load (keypointrcnn_rhpe_<model>.pth)')
    parser.add_argument('--mode',  type=str, default='full',
                        choices=['full', 'gt_crop', 'det_crop'],
                        help='full: whole image | gt_crop: crop with gt bbox | det_crop: crop with predicted bbox')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Split: {args.split} | Model: {args.model} | Mode: {args.mode} | Visualize: {args.visualize}")

    config  = SPLIT_CONFIG[args.split]
    dataset = RHPEKeypointEvalDataset(config['image_dir'], config['json_path'])

    # Load keypoint model — checkpoint depends on how it was trained (--model)
    kpt_checkpoint = f"{KPT_CHECKPOINT_DIR}/keypointrcnn_rhpe_{args.model}.pth"
    kpt_model = get_kpt_model().to(device)
    kpt_model.load_state_dict(torch.load(kpt_checkpoint, map_location=device, weights_only=True))
    kpt_model.eval()
    print(f"Keypoint model loaded from {kpt_checkpoint}")

    # Load detection model if needed for det_crop
    det_model = None
    if args.mode == 'det_crop':
        det_model = get_det_model().to(device)
        det_model.load_state_dict(torch.load(DET_CHECKPOINT, map_location=device, weights_only=True))
        det_model.eval()
        print("Detection model loaded successfully")

    output_dir = f"/data01/giacomelli/experiments/keypoints/logs/{args.split}/model_{args.model}_mode_{args.mode}"
    os.makedirs(output_dir, exist_ok=True)

    to_tensor   = T.ToTensor()
    predictions = []

    with torch.no_grad():
        for i, (img_pil, coco_id) in enumerate(dataset):
            offset = (0., 0.)  # (dx, dy) to shift predictions back to original coords

            if args.mode == 'full':
                img_tensor = to_tensor(img_pil).to(device)
                pred       = kpt_model([img_tensor])[0]

            elif args.mode == 'gt_crop':
                x, y, w, h = dataset.id_to_ann[coco_id]['bbox']
                img_crop   = img_pil.crop((x, y, x + w, y + h))
                img_tensor = to_tensor(img_crop).to(device)
                pred       = kpt_model([img_tensor])[0]
                offset     = (x, y)

            elif args.mode == 'det_crop':
                img_tensor = to_tensor(img_pil).to(device)
                det_pred   = det_model([img_tensor])[0]

                if len(det_pred['boxes']) == 0:
                    continue

                x1, y1, x2, y2 = det_pred['boxes'][0].cpu().numpy()
                img_crop   = img_pil.crop((x1, y1, x2, y2))
                img_tensor = to_tensor(img_crop).to(device)
                pred       = kpt_model([img_tensor])[0]
                offset     = (x1, y1)

            if len(pred['scores']) == 0:
                continue

            kpts  = pred['keypoints'][0].cpu().numpy()
            kpts[:, 0] += offset[0]
            kpts[:, 1] += offset[1]
            score = float(pred['scores'][0].cpu())

            flat_kpts = []
            for (x, y, _v) in kpts:
                flat_kpts.extend([float(x), float(y), 1])

            predictions.append({
                "image_id":    int(coco_id),
                "category_id": CATEGORY_ID,
                "keypoints":   flat_kpts,
                "score":       score,
            })

            if args.visualize and i < 3:
                draw = ImageDraw.Draw(img_pil)

                for (x, y, _v) in kpts:
                    draw.ellipse([x - 5, y - 5, x + 5, y + 5], outline='red', width=3)

                gt_kpts = dataset.id_to_ann[coco_id]['keypoints']
                for k in range(NUM_KEYPOINTS):
                    gx, gy, gv = gt_kpts[3*k], gt_kpts[3*k+1], gt_kpts[3*k+2]
                    if gv > 0:
                        draw.ellipse([gx-5, gy-5, gx+5, gy+5], outline='green', width=3)

                out_path = f"{output_dir}/sample_{i+1}.png"
                img_pil.save(out_path)
                print(f"  [Visualization] Saved to: {out_path}")

            if i % 200 == 0:
                print(f"  Processed {i}/{len(dataset)} images...")

    # ── Save predictions and run COCOeval ────────────────────────────────────
    pred_path = f"{output_dir}/{args.split}_predictions.json"
    with open(pred_path, 'w') as f:
        json.dump(predictions, f)

    cocoGt   = COCO(config['json_path'])
    cocoDt   = cocoGt.loadRes(pred_path)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='keypoints')
    cocoEval.params.kpt_oks_sigmas = np.full(NUM_KEYPOINTS, KPT_OKS_SIGMA)
    cocoEval.params.imgIds = sorted(cocoGt.getImgIds())
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    stats = cocoEval.stats
    print(f"\n=== Results — {args.split.upper()} | Model: {args.model} | Mode: {args.mode} ===")
    print(f"mAP@[.5:.95]: {stats[0]:.4f}")
    print(f"mAP@.50:      {stats[1]:.4f}")
    print(f"mAP@.75:      {stats[2]:.4f}")
    print(f"mAR@[.5:.95]: {stats[5]:.4f}")


if __name__ == '__main__':
    main()