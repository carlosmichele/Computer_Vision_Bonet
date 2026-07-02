import os
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

BASE = "/data01/giacomelli/data"
ANN  = f"{BASE}/RHPE_Annotations"

SEED          = 42
NUM_KEYPOINTS = 17

random.seed(SEED)
torch.manual_seed(SEED)

# ── Stratified Sampling ───────────────────────────────────────────────────
def stratified_sample(json_path, csv_path, n_samples, split_name):
    with open(json_path) as f:
        coco = json.load(f)

    id_to_file = {img['id']: img['file_name'] for img in coco['images']}
    id_to_ann  = {}
    for ann in coco['annotations']:
        if ann['image_id'] not in id_to_ann:
            id_to_ann[ann['image_id']] = ann

    df      = pd.read_csv(csv_path)
    age_col = 'Boneage' if 'Boneage' in df.columns else 'Chronological'
    df['bin'] = (df[age_col] // 24).astype(int)

    sampled_ids = []
    for bin_val, group in df.groupby('bin'):
        n       = max(1, round(n_samples * len(group) / len(df)))
        sampled = group.sample(min(n, len(group)), random_state=SEED)
        sampled_ids.extend(sampled['ID'].tolist())

    sampled_ids = sampled_ids[:n_samples]
    print(f"{split_name}: {len(sampled_ids)} images sampled")
    return sampled_ids, id_to_file, id_to_ann

# ── Dataset ───────────────────────────────────────────────────────────────
class RHPEKeypointDataset(Dataset):
    def __init__(self, image_dir, sampled_ids, id_to_file, id_to_ann, transforms=None):
        self.image_dir  = image_dir
        self.transforms = transforms
        self.samples    = []

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
        img = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")

        x, y, w, h = ann['bbox']

        # Crop image using ground truth bbox
        img = img.crop((x, y, x + w, y + h))

        # Bbox in crop coordinates: the whole crop
        boxes  = torch.tensor([[0., 0., float(w), float(h)]], dtype=torch.float32)
        labels = torch.ones((1,), dtype=torch.int64)

        # Keypoints: adjust to crop coordinates by subtracting bbox offset
        kps       = ann['keypoints']
        keypoints = torch.tensor(kps, dtype=torch.float32).reshape(1, NUM_KEYPOINTS, 3)
        keypoints[0, :, 0] -= x
        keypoints[0, :, 1] -= y

        target = {
            'boxes':     boxes,
            'labels':    labels,
            'keypoints': keypoints,
            'image_id':  torch.tensor([idx])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

# ── Model ─────────────────────────────────────────────────────────────────
def get_model(num_keypoints=NUM_KEYPOINTS):
    model       = keypointrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_features, num_keypoints)
    return model

# ── Transforms ────────────────────────────────────────────────────────────
transform = T.Compose([T.ToTensor()])

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    # Sampling
    train_ids, id_to_file_train, id_to_ann_train = stratified_sample(
        f"{ANN}/RHPE_anatomical_ROIs_train.json",
        f"{ANN}/RHPE_Boneage_train.csv", 250, "TRAIN")

    val_ids, id_to_file_val, id_to_ann_val = stratified_sample(
        f"{ANN}/RHPE_anatomical_ROIs_val.json",
        f"{ANN}/RHPE_Boneage_val.csv", 50, "VAL")

    # Datasets and DataLoaders
    train_dataset = RHPEKeypointDataset(
        f"{BASE}/RHPE_train", train_ids, id_to_file_train, id_to_ann_train, transform)
    val_dataset = RHPEKeypointDataset(
        f"{BASE}/RHPE_val", val_ids, id_to_file_val, id_to_ann_val, transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                              collate_fn=lambda x: tuple(zip(*x)))
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False,
                              collate_fn=lambda x: tuple(zip(*x)))

    # Model and optimizer — SGD with standard hyperparameters
    model     = get_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # as indicated in the BoNet paper for keypoint estimation:
    num_epochs = 20

    save_path = "/data01/giacomelli/experiments/keypoints/checkpoints/keypointrcnn_rhpe.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_val_map = -1.0

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0

        for imgs, targets in train_loader:
            imgs    = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses    = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_train_loss += losses.item()

        avg_train_loss = epoch_train_loss / len(train_loader)

        # Validation — mAP on bbox as proxy for best model selection
        model.eval()
        metric       = MeanAveragePrecision()
        preds_list   = []
        targets_list = []

        with torch.no_grad():
            for val_imgs, val_targets in val_loader:
                val_imgs_gpu = [img.to(device) for img in val_imgs]
                preds        = model(val_imgs_gpu)

                for pred, target in zip(preds, val_targets):
                    preds_list.append({
                        'boxes':  pred['boxes'].cpu(),
                        'scores': pred['scores'].cpu(),
                        'labels': pred['labels'].cpu(),
                    })
                    targets_list.append({
                        'boxes':  target['boxes'],
                        'labels': target['labels'],
                    })

        metric.update(preds_list, targets_list)
        val_results = metric.compute()
        val_map     = val_results['map'].item()
        val_map_50  = val_results['map_50'].item()

        print(f"Epoch {epoch+1}/{num_epochs} ── "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val mAP@.50: {val_map_50:.4f} | "
              f"Val mAP@[.5:.95]: {val_map:.4f}")

        if val_map > best_val_map:
            best_val_map = val_map
            torch.save(model.state_dict(), save_path)
            print(f"  => New best model saved with mAP@[.5:.95]: {best_val_map:.4f}")

    print(f"Training complete. Best Validation mAP@[.5:.95] achieved: {best_val_map:.4f}")


if __name__ == '__main__':
    main()