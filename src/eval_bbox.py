import os
import json
import argparse
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

BASE       = "/data01/giacomelli/data"
ANN        = f"{BASE}/RHPE_Annotations"
CHECKPOINT = "/data01/giacomelli/experiments/bbox_detection/checkpoints/fasterrcnn_rhpe.pth"

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
class RHPEDetectionDataset(Dataset):
    def __init__(self, image_dir, json_path, transforms=None):
        self.image_dir  = image_dir
        self.transforms = transforms

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
        ann     = self.id_to_ann[coco_id]

        img = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")

        x, y, w, h = ann['bbox']
        boxes  = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)
        labels = torch.ones((1,), dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target
    
# ── Model ────────────────────────────────────────────────────────────────────
def get_model(num_classes=2):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'])
    parser.add_argument('--visualize', action='store_true', help='Save sample plots with bounding boxes')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Split: {args.split} | Visualize: {args.visualize}")

    config  = SPLIT_CONFIG[args.split]
    dataset = RHPEDetectionDataset(
        config['image_dir'], config['json_path'],
        transforms=T.Compose([T.ToTensor()]))

    loader = DataLoader(dataset, batch_size=4, shuffle=False,
                        collate_fn=lambda x: tuple(zip(*x)),
                        num_workers=4)

    model = get_model().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully")

    metric      = MeanAveragePrecision()
    preds_list  = []
    targets_list = []

    output_dir = f"/data01/giacomelli/experiments/bbox_detection/logs/{args.split}"
    if args.visualize:
        os.makedirs(output_dir, exist_ok=True)
        from PIL import ImageDraw

    with torch.no_grad():
        for i, (imgs, targets) in enumerate(loader):
            imgs_gpu = [img.to(device) for img in imgs]
            preds    = model(imgs_gpu)

            for j, (pred, target) in enumerate(zip(preds, targets)):
                preds_list.append({
                    'boxes':  pred['boxes'].cpu(),
                    'scores': pred['scores'].cpu(),
                    'labels': pred['labels'].cpu()
                })
                targets_list.append({
                    'boxes':  target['boxes'],
                    'labels': target['labels']
                })

                if args.visualize and (i * loader.batch_size + j) < 3:
                    img_pil = T.ToPILImage()(imgs[j])
                    draw = ImageDraw.Draw(img_pil)

                    if len(pred['boxes']) > 0:
                        box = pred['boxes'][0].cpu().numpy()
                        draw.rectangle(box.tolist(), outline='red', width=4)

                    gt_box = target['boxes'][0].numpy()
                    draw.rectangle(gt_box.tolist(), outline='green', width=4)

                    global_idx = i * loader.batch_size + j
                    out_path = f"{output_dir}/sample_{global_idx + 1}.png"
                    img_pil.save(out_path)
                    print(f"  [Visualization] Saved sample image to: {out_path}")

            if i % 50 == 0:
                print(f"  Processed {i * loader.batch_size}/{len(dataset)} images...")

    metric.update(preds_list, targets_list)
    results = metric.compute()

    print(f"\n=== Results — {args.split.upper()} ===")
    print(f"mAP@[.5:.95]: {results['map'].item():.4f}")
    print(f"mAP@.50:      {results['map_50'].item():.4f}")
    print(f"mAP@.75:      {results['map_75'].item():.4f}")

if __name__ == '__main__':
    main()