import os
import json
import torch
from PIL import Image, ImageDraw
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

BASE = "/data01/giacomelli/data"
ANN  = f"{BASE}/RHPE_Annotations"
CHECKPOINT = "/data01/giacomelli/experiments/bbox_detection/checkpoints/fasterrcnn_rhpe.pth"
OUTPUT_DIR = "/data01/giacomelli/experiments/bbox_detection/logs/test"

# ── 1. Dataset ───────────────────────────────────────────────────────────────
class RHPETestDataset(Dataset):
    def __init__(self, image_dir, json_path, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms

        with open(json_path) as f:
            coco = json.load(f)

        self.id_to_file = {img['id']: img['file_name'] for img in coco['images']}
        self.id_to_ann  = {}
        for ann in coco['annotations']:
            if ann['image_id'] not in self.id_to_ann:
                self.id_to_ann[ann['image_id']] = ann

        self.image_ids = list(self.id_to_file.keys())

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

        return img, target, fname


# ── 2. Model ─────────────────────────────────────────────────────────────────
def get_model(num_classes=2):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ── 3. Main ──────────────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    transform = T.Compose([T.ToTensor()])

    test_dataset = RHPETestDataset(
        f"{BASE}/RHPE_test",
        f"{ANN}/RHPE_anatomical_ROIs_test.json",
        transforms=transform)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             collate_fn=lambda x: tuple(zip(*x)))

    # Load model
    model = get_model().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))    
    model.eval()
    print("Model loaded successfully")

    # mAP metric
    metric = MeanAveragePrecision()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    preds_list  = []
    targets_list = []

    with torch.no_grad():
        for i, (imgs, targets, fnames) in enumerate(test_loader):
            img_tensor = imgs[0].to(device)
            pred = model([img_tensor])[0]

            # Accumulate for mAP
            preds_list.append({
                'boxes':  pred['boxes'].cpu(),
                'scores': pred['scores'].cpu(),
                'labels': pred['labels'].cpu()
            })
            targets_list.append({
                'boxes':  targets[0]['boxes'],
                'labels': targets[0]['labels']
            })

            # Visualize first 3
            if i < 3:
                img_pil = T.ToPILImage()(imgs[0])
                draw = ImageDraw.Draw(img_pil)

                if len(pred['boxes']) > 0:
                    box = pred['boxes'][0].cpu().numpy()
                    draw.rectangle(box.tolist(), outline='red', width=4)

                gt_box = targets[0]['boxes'][0].numpy()
                draw.rectangle(gt_box.tolist(), outline='green', width=4)

                img_pil.save(f"{OUTPUT_DIR}/test_{i+1}_{fnames[0]}")
                print(f"Saved visualization: test_{i+1}_{fnames[0]}")

    # Compute mAP
    metric.update(preds_list, targets_list)
    results = metric.compute()
    print(f"\n=== Test Results ===")
    print(f"mAP@[.5:.95]: {results['map'].item():.4f}")
    print(f"mAP@.50:      {results['map_50'].item():.4f}")
    print(f"mAP@.75:      {results['map_75'].item():.4f}")


if __name__ == '__main__':
    main()