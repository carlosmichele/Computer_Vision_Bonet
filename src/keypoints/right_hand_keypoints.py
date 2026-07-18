"""
Exploratory script: predict keypoints on both hands of a full radiograph.
No ground truth exists for the right hand in RHPE, so this is always a
qualitative visual check only: no metrics are computed.

Four strategies were tried, selectable with --version:

v1: detection with 'full' model on the whole image (top-2 detections),
    keypoints with 'mirrored' model on each detected crop.
    -> 'full' rarely finds a second hand (never trained with 2 bboxes
    per image); spurious low-score duplicate detections of the same
    left hand were observed instead of a true right-hand detection.

v2: image split into left/right halves, detection with 'full' model
    run separately on each half, keypoints with 'mirrored' model on
    each detected crop.
    -> inconsistent: right hand detected with good score in some
    images, very low score or no detection at all in others.

v3: image split into left/right halves, BOTH detection and keypoints
    done with the 'mirrored' model on each half.
    -> detection now works reliably on both halves (high scores), but
    keypoints come out wrong on both hands. Cause: 'mirrored' was
    trained only on tight ground-truth crops (bbox target always the
    full crop), so its detector never learned to precisely localize a
    hand within a wider context (a half-image has more background than
    a tight crop).

v4: single forward pass on the whole image using the 'mirrored_full'
    model (trained on whole images + their mirrored version, bbox and
    keypoints mirrored using the full image width). No splitting or
    two-stage pipeline needed — detection and keypoints for both hands
    come directly from one model call.
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
SCORE_THRESHOLD = 0.1  # low threshold for v1/v2/v3 debug visibility
SCORE_THRESHOLD_V4 = 0.3
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


def load_model(name, device):
    model = get_model().to(device)
    model.load_state_dict(torch.load(
        f"{KPT_CHECKPOINT_DIR}/keypointrcnn_rhpe_{name}.pth",
        map_location=device, weights_only=True))
    model.eval()
    print(f"Model '{name}' loaded")
    return model


# ── v1: full-image detection (top-2) + mirrored keypoints on each crop ──────
def run_v1(img_pil, det_model, kpt_model, to_tensor, device):
    img_tensor = to_tensor(img_pil).to(device)
    det_pred   = det_model([img_tensor])[0]

    keep   = det_pred['scores'] > SCORE_THRESHOLD
    boxes  = det_pred['boxes'][keep][:MAX_HANDS].cpu().numpy()
    scores = det_pred['scores'][keep][:MAX_HANDS].cpu().numpy()
    print(f"    All scores: {det_pred['scores'].cpu().numpy()[:5]}")

    results = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        crop = img_pil.crop((x1, y1, x2, y2))
        crop_tensor = to_tensor(crop).to(device)
        kpt_pred = kpt_model([crop_tensor])[0]

        kpts = None
        if len(kpt_pred['scores']) > 0:
            kpts = kpt_pred['keypoints'][0].cpu().numpy()
            kpts[:, 0] += x1
            kpts[:, 1] += y1

        results.append(((x1, y1, x2, y2), kpts, score))
    return results


# ── v2: half-split detection (full) + mirrored keypoints on each crop ───────
def run_v2(img_pil, det_model, kpt_model, to_tensor, device):
    width, height = img_pil.size
    mid = width // 2
    halves = [("LEFT", img_pil.crop((0, 0, mid, height)), 0),
              ("RIGHT", img_pil.crop((mid, 0, width, height)), mid)]

    results = []
    for label, half_img, x_offset in halves:
        half_tensor = to_tensor(half_img).to(device)
        det_pred    = det_model([half_tensor])[0]

        if len(det_pred['scores']) == 0:
            print(f"    [{label}] no detections at all")
            continue
        top_score = float(det_pred['scores'][0].cpu())
        if top_score < SCORE_THRESHOLD:
            print(f"    [{label}] best score {top_score:.4f} below threshold {SCORE_THRESHOLD}")
            continue
        print(f"    [{label}] detected with score {top_score:.4f}")

        x1, y1, x2, y2 = det_pred['boxes'][0].cpu().numpy()
        crop = half_img.crop((x1, y1, x2, y2))
        crop_tensor = to_tensor(crop).to(device)
        kpt_pred = kpt_model([crop_tensor])[0]

        kpts = None
        if len(kpt_pred['scores']) > 0:
            kpts = kpt_pred['keypoints'][0].cpu().numpy()
            kpts[:, 0] += x1 + x_offset
            kpts[:, 1] += y1

        box_original = (x1 + x_offset, y1, x2 + x_offset, y2)
        results.append((box_original, kpts, top_score))
    return results


# ── v3: half-split, single model (mirrored) for both detection+keypoints ────
def run_v3(img_pil, mirrored_model, to_tensor, device):
    width, height = img_pil.size
    mid = width // 2
    halves = [("LEFT", img_pil.crop((0, 0, mid, height)), 0),
              ("RIGHT", img_pil.crop((mid, 0, width, height)), mid)]

    results = []
    for label, half_img, x_offset in halves:
        half_tensor = to_tensor(half_img).to(device)
        pred        = mirrored_model([half_tensor])[0]

        if len(pred['scores']) == 0:
            print(f"    [{label}] no detections at all")
            continue
        top_score = float(pred['scores'][0].cpu())
        if top_score < SCORE_THRESHOLD:
            print(f"    [{label}] best score {top_score:.4f} below threshold {SCORE_THRESHOLD}")
            continue
        print(f"    [{label}] detected with score {top_score:.4f}")

        x1, y1, x2, y2 = pred['boxes'][0].cpu().numpy()
        kpts = pred['keypoints'][0].cpu().numpy()
        kpts[:, 0] += x_offset

        box_original = (x1 + x_offset, y1, x2 + x_offset, y2)
        results.append((box_original, kpts, top_score))
    return results


# ── v4: single pass, whole image, mirrored_full model ────────────────────────
def run_v4(img_pil, mirrored_full_model, to_tensor, device):
    img_tensor = to_tensor(img_pil).to(device)
    pred = mirrored_full_model([img_tensor])[0]

    scores_preview = pred['scores'].cpu().numpy()[:5]
    print(f"    All scores: {scores_preview}")

    keep     = pred['scores'] > SCORE_THRESHOLD_V4
    boxes    = pred['boxes'][keep][:MAX_HANDS].cpu().numpy()
    kpts_all = pred['keypoints'][keep][:MAX_HANDS].cpu().numpy()
    scores   = pred['scores'][keep][:MAX_HANDS].cpu().numpy()

    results = []
    for box, kpts, score in zip(boxes, kpts_all, scores):
        x1, y1, x2, y2 = box
        results.append(((x1, y1, x2, y2), kpts, score))
    return results


def draw_results(img_pil, results):
    vis  = img_pil.copy()
    draw = ImageDraw.Draw(vis)
    colors_box = ['yellow', 'cyan']
    colors_kpt = ['red', 'lime']

    for hand_idx, (box, kpts, score) in enumerate(results):
        x1, y1, x2, y2 = box
        c = colors_box[hand_idx % 2]
        draw.rectangle([x1, y1, x2, y2], outline=c, width=4)
        draw.text((x1, max(0, y1 - 15)), f"hand {hand_idx} ({score:.2f})", fill=c)

        if kpts is not None:
            for (kx, ky, _v) in kpts:
                draw.ellipse([kx - 5, ky - 5, kx + 5, ky + 5], outline=colors_kpt[hand_idx % 2], width=3)

    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--version', type=str, required=True, choices=['v1', 'v2', 'v3', 'v4'])
    parser.add_argument('--num_images', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Split: {args.split} | Version: {args.version}")

    config = SPLIT_CONFIG[args.split]

    with open(config['json_path']) as f:
        coco = json.load(f)
    id_to_file = {img['id']: img['file_name'] for img in coco['images']}
    image_ids  = list(id_to_file.keys())[:args.num_images]

    to_tensor = T.ToTensor()

    # Load only the models needed for the selected version
    if args.version == 'v1':
        det_model = load_model('full', device)
        kpt_model = load_model('mirrored', device)
    elif args.version == 'v2':
        det_model = load_model('full', device)
        kpt_model = load_model('mirrored', device)
    elif args.version == 'v3':
        mirrored_model = load_model('mirrored', device)
    elif args.version == 'v4':
        mirrored_full_model = load_model('mirrored_full', device)

    output_dir = f"/data01/giacomelli/experiments/keypoints/logs/explore_right_hand_{args.version}/{args.split}"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, coco_id in enumerate(image_ids):
            fname   = id_to_file[coco_id]
            img_pil = Image.open(os.path.join(config['image_dir'], fname)).convert("RGB")

            print(f"\n[{fname}]")
            if args.version == 'v1':
                results = run_v1(img_pil, det_model, kpt_model, to_tensor, device)
            elif args.version == 'v2':
                results = run_v2(img_pil, det_model, kpt_model, to_tensor, device)
            elif args.version == 'v3':
                results = run_v3(img_pil, mirrored_model, to_tensor, device)
            elif args.version == 'v4':
                results = run_v4(img_pil, mirrored_full_model, to_tensor, device)

            print(f"  Detected {len(results)} hand(s)")

            vis = draw_results(img_pil, results)
            out_path = f"{output_dir}/sample_{idx+1}_{fname}"
            vis.save(out_path)
            print(f"  Saved visualization: {out_path}")

    print(f"\nDone. Visualizations saved to {output_dir}")
    print("NOTE: no ground truth available for the right hand — visual inspection only.")


if __name__ == '__main__':
    main()