import os
import json
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader

BASE = "/data01/giacomelli/data"
ANN  = f"{BASE}/RHPE_Annotations"

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ── 1. Stratified Sampling ───────────────────────────────────────────────────
def stratified_sample(json_path, csv_path, n_samples, split_name):
    with open(json_path) as f:
        coco = json.load(f)

    id_to_file = {img['id']: img['file_name'] for img in coco['images']}
    id_to_ann = {}
    for ann in coco['annotations']:
        if ann['image_id'] not in id_to_ann:
            id_to_ann[ann['image_id']] = ann

    df = pd.read_csv(csv_path)
    age_col = 'Boneage' if 'Boneage' in df.columns else 'Chronological'
    df['bin'] = (df[age_col] // 24).astype(int)

    sampled_ids = []
    for bin_val, group in df.groupby('bin'):
        n = max(1, round(n_samples * len(group) / len(df)))
        sampled = group.sample(min(n, len(group)), random_state=SEED)
        sampled_ids.extend(sampled['ID'].tolist())

    sampled_ids = sampled_ids[:n_samples]
    print(f"{split_name}: {len(sampled_ids)} images sampled")
    return sampled_ids, id_to_file, id_to_ann


# ── 2. Dataset ───────────────────────────────────────────────────────────────
class RHPEDetectionDataset(Dataset):
    def __init__(self, image_dir, sampled_ids, id_to_file, id_to_ann, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.samples = []

        for img_id in sampled_ids:
            for coco_id, fname in id_to_file.items():
                file_id = int(os.path.splitext(fname)[0])
                if file_id == img_id and coco_id in id_to_ann:
                    self.samples.append((fname, id_to_ann[coco_id]))
                    break

        print(f"Dataset: {len(self.samples)} valid samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, ann = self.samples[idx]
        img_path = os.path.join(self.image_dir, fname)
        img = Image.open(img_path).convert("RGB")

        # Convert bbox from COCO format [x, y, w, h] to [x1, y1, x2, y2]
        x, y, w, h = ann['bbox']
        boxes  = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)
        labels = torch.ones((1,), dtype=torch.int64)  # class 1 = hand

        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx])}

        if self.transforms:
            img = self.transforms(img)

        return img, target


# ── 3. Model ─────────────────────────────────────────────────────────────────
def get_model(num_classes=2):  # background + hand
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ── 4. Transforms ────────────────────────────────────────────────────────────
transform = T.Compose([T.ToTensor()])


# ── 5. Main ──────────────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Sampling
    train_ids, id_to_file_train, id_to_ann_train = stratified_sample(
        f"{ANN}/RHPE_anatomical_ROIs_train.json",
        f"{ANN}/RHPE_Boneage_train.csv", 250, "TRAIN")

    val_ids, id_to_file_val, id_to_ann_val = stratified_sample(
        f"{ANN}/RHPE_anatomical_ROIs_val.json",
        f"{ANN}/RHPE_Boneage_val.csv", 50, "VAL")

    # Datasets and DataLoaders
    train_dataset = RHPEDetectionDataset(
        f"{BASE}/RHPE_train", train_ids, id_to_file_train, id_to_ann_train, transform)
    val_dataset = RHPEDetectionDataset(
        f"{BASE}/RHPE_val", val_ids, id_to_file_val, id_to_ann_val, transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                              collate_fn=lambda x: tuple(zip(*x)))
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False,
                              collate_fn=lambda x: tuple(zip(*x)))

    # Model and optimizer — constant lr as in the paper (2.5e-3)
    model = get_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=2.5e-3,
                                momentum=0.9, weight_decay=0.0005)

    # Training loop — 5000 iterations as in the paper
    total_iterations = 5000
    batch_size = 4
    iters_per_epoch = len(train_loader)
    num_epochs = total_iterations // iters_per_epoch + 1

    print(f"Iterations per epoch: {iters_per_epoch}")
    print(f"Total epochs to reach ~5000 iterations: {num_epochs}")

    iteration = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for imgs, targets in train_loader:
            if iteration >= total_iterations:
                break

            imgs    = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            iteration += 1

        if iteration >= total_iterations:
            break

        print(f"Epoch {epoch+1} (iter {iteration}/{total_iterations}) — Loss: {epoch_loss/iters_per_epoch:.4f}")

    print(f"Training complete at iteration {iteration}")

    # Save model
    save_path = "/data01/giacomelli/experiments/bbox_detection/checkpoints/fasterrcnn_rhpe.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

    # Visualize predictions on 3 val images
    model.eval()
    os.makedirs("/data01/giacomelli/experiments/bbox_detection/logs", exist_ok=True)
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(val_loader):
            if i >= 3:
                break
            img_tensor = imgs[0].to(device)
            pred = model([img_tensor])[0]

            img_pil = T.ToPILImage()(imgs[0])
            draw = ImageDraw.Draw(img_pil)

            # Predicted bbox (best scoring)
            if len(pred['boxes']) > 0:
                box = pred['boxes'][0].cpu().numpy()
                draw.rectangle(box.tolist(), outline='red', width=4)

            # Ground truth bbox
            gt_box = targets[0]['boxes'][0].numpy()
            draw.rectangle(gt_box.tolist(), outline='green', width=4)

            out_path = f"/data01/giacomelli/experiments/bbox_detection/logs/val/val_{i+1}.png"
            img_pil.save(out_path)
            print(f"Saved visualization: {out_path}")


if __name__ == '__main__':
    main()