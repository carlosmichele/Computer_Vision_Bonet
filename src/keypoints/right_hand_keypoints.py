"""
Exploratory script: detect both hands directly in a full radiograph using
the 'mirrored_full' model (trained on whole images + their horizontally
mirrored version, with bbox and keypoints mirrored accordingly). Since
this model has learned to localize AND predict keypoints for a hand in
either orientation within a full-image context, no separate detection
model or image splitting is needed — we simply take the top-N detections
directly from a single forward pass.

Both hands' keypoints are drawn on the same original image.

No ground truth exists for the right hand in RHPE, so this is a
qualitative visual check only — no metrics are computed.
"""

import os
import json
import argparse
import torch
from PIL import Image, ImageDraw
from torchvision import transforms as T
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

BASE = "/data01/giacomelli/data"
ANN  = f"{BASE}/RHPE_Annotations"
KPT_CHECKPOINT_DIR = "/data01/giacomelli/experiments/keypoints/checkpoints"

NUM_KEYPOINTS = 17
SCORE_THRESHOLD = 0.3
MAX_HANDS = 2

SPLIT_CONFIG = {
    "train": {"image_dir": f"{BASE}/RHPE_train", "json_path": f"{ANN}/RHPE_anatomical_ROIs_train.json"},
    "val":   {"image_dir": f"{BASE}/RHPE_val",   "json_path": f"{ANN}/RHPE_anatomical_ROIs_val.json"},
    "test":  {"image_dir": f"{BASE}/RHPE_test",  "json_path": f"{ANN}/RHPE_anatomical_ROIs_test.json"},
}


def get_model(num_keypoints=NUM_KEYPOINTS):
    model       = keypointrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_features, num_keypoints)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--num_images', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Split: {args.split}")

    config = SPLIT_CONFIG[args.split]

    with open(config['json_path']) as f:
        coco = json.load(f)
    id_to_file = {img['id']: img['file_name'] for img in coco['images']}
    image_ids  = list(id_to_file.keys())[:args.num_images]

    model = get_model().to(device)
    model.load_state_dict(torch.load(
        f"{KPT_CHECKPOINT_DIR}/keypointrcnn_rhpe_mirrored_full.pth",
        map_location=device, weights_only=True))
    model.eval()
    print("Mirrored_full model loaded (single-pass detection + keypoints)")

    output_dir = f"/data01/giacomelli/experiments/keypoints/logs/explore_right_hand_v2/{args.split}"
    os.makedirs(output_dir, exist_ok=True)

    to_tensor = T.ToTensor()

    with torch.no_grad():
        for idx, coco_id in enumerate(image_ids):
            fname   = id_to_file[coco_id]
            img_pil = Image.open(os.path.join(config['image_dir'], fname)).convert("RGB")

            img_tensor = to_tensor(img_pil).to(device)
            pred = model([img_tensor])[0]

            scores = pred['scores'].cpu().numpy()
            print(f"\n[{fname}] All scores: {scores[:5]}")  # print top 5 for inspection

            keep = pred['scores'] > SCORE_THRESHOLD
            boxes  = pred['boxes'][keep][:MAX_HANDS].cpu().numpy()
            kpts_all = pred['keypoints'][keep][:MAX_HANDS].cpu().numpy()
            kept_scores = pred['scores'][keep][:MAX_HANDS].cpu().numpy()

            vis  = img_pil.copy()
            draw = ImageDraw.Draw(vis)

            colors_box = ['yellow', 'cyan']
            colors_kpt = ['red', 'lime']

            for hand_idx, (box, kpts, score) in enumerate(zip(boxes, kpts_all, kept_scores)):
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline=colors_box[hand_idx % 2], width=4)
                draw.text((x1, max(0, y1 - 15)), f"hand {hand_idx} ({score:.2f})", fill=colors_box[hand_idx % 2])

                for (kx, ky, _v) in kpts:
                    draw.ellipse([kx - 5, ky - 5, kx + 5, ky + 5], outline=colors_kpt[hand_idx % 2], width=3)

            print(f"  Detected {len(boxes)} hand(s) above threshold {SCORE_THRESHOLD}")

            out_path = f"{output_dir}/sample_{idx+1}_{fname}"
            vis.save(out_path)
            print(f"  Saved visualization: {out_path}")

    print(f"\nDone. Visualizations saved to {output_dir}")
    print("NOTE: no ground truth available for the right hand — visual inspection only.")


if __name__ == '__main__':
    main()