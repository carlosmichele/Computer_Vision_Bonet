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

# Minimum confidence to be considered a candidate for the "leftmost box"
# selection in 'full' mode.
LEFT_HAND_SCORE_THRESHOLD = 0.5

# Per-keypoint OKS sigmas as reported in the BoNet supplementary material
# (Escobar et al., MICCAI 2019), computed from the variance across
# different trained annotators labeling the same image. Order assumed to
# match the RHPE keypoint annotation order: f1_0, f1_1, f1_2 (finger 1),
# f2_0-2, f3_0-2, f4_0-2, f5_0-2 (fingers 2-5), anchor_0, anchor_1.
# These are roughly an order of magnitude stricter than COCO's standard
# human-body sigmas (0.025-0.107), reflecting how precisely each hand
# landmark can be localized.
KPT_OKS_SIGMAS = np.array([
    0.0110, 0.0107, 0.0090,   # f1
    0.0076, 0.0074, 0.0071,   # f2
    0.0039, 0.0039, 0.0040,   # f3
    0.0070, 0.0071, 0.0071,   # f4
    0.0120, 0.0120, 0.0120,   # f5
    0.0090, 0.0080,           # anchor
])

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
    parser.add_argument('--model', type=str, required=True,
                        choices=['full', 'gt_crop', 'mirrored', 'mirrored_full'],
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

            # In 'full' mode the model may detect both hands (mirrored_full
            # was trained on both orientations); we always want the left
            # hand here, since that's what the ground truth annotates.
            # Among detections above LEFT_HAND_SCORE_THRESHOLD, pick the one
            # whose box is leftmost (smallest x1) — filtering by score first
            # avoids picking a low-confidence spurious detection (e.g. a
            # label/marker in the image) just because it happens to sit
            # further left than the real hand. If nothing clears the
            # threshold, fall back to the single top-scoring detection
            # (predictions are already sorted by descending score).
            if args.mode == 'full' and len(pred['boxes']) > 1:
                scores_np = pred['scores'].cpu().numpy()
                keep = np.where(scores_np > LEFT_HAND_SCORE_THRESHOLD)[0]
                if len(keep) == 0:
                    best_idx = 0
                else:
                    x1_coords = pred['boxes'][keep, 0].cpu().numpy()
                    best_idx  = int(keep[x1_coords.argmin()])
            else:
                best_idx = 0

            kpts  = pred['keypoints'][best_idx].cpu().numpy()
            kpts[:, 0] += offset[0]
            kpts[:, 1] += offset[1]
            score = float(pred['scores'][best_idx].cpu())

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
    cocoEval.params.kpt_oks_sigmas = KPT_OKS_SIGMAS
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

    all_metrics_path = "/data01/giacomelli/experiments/keypoints/logs/all_metrics.json"
    key = f"{args.split}_{args.model}_{args.mode}"

    if os.path.exists(all_metrics_path):
        with open(all_metrics_path) as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    all_metrics[key] = {
        "split":         args.split,
        "model":         args.model,
        "mode":          args.mode,
        "map_50_95":     round(float(stats[0]), 4),
        "map_50":        round(float(stats[1]), 4),
        "map_75":        round(float(stats[2]), 4),
        "mar_50_95":     round(float(stats[5]), 4),
    }

    with open(all_metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to {all_metrics_path} (key: {key})")


if __name__ == '__main__':
    main()