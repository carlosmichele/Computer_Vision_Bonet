import os
import sys
import random
import argparse
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms as T
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rhpe_data import stratified_sample, load_all_ids, match_samples

BASE = "/data01/giacomelli/data"
ANN  = f"{BASE}/RHPE_Annotations"

SEED          = 42
NUM_KEYPOINTS = 17

random.seed(SEED)
torch.manual_seed(SEED)

# ── Dataset ───────────────────────────────────────────────────────────────
class RHPEKeypointDataset(Dataset):
    """
    mode='full':     the whole image is used as input; the bbox in the
                      target keeps its original (full-image) coordinates.
    mode='gt_crop':   the image is cropped to the ground truth bbox before
                      being fed to the model; bbox and keypoints are
                      shifted to crop-local coordinates.
    mode='mirrored':  same as 'gt_crop', but the dataset is doubled: for
                      every sample, a horizontally flipped version is also
                      added (synthetic right hand, with keypoints mirrored
                      accordingly). Ground truth keypoints only exist for
                      the left hand in RHPE, so mirroring is the only way
                      to obtain training data for the right hand.
    mode='mirrored_full': same idea as 'mirrored', but applied to the whole
                      image instead of the crop. For every sample, a
                      horizontally flipped version of the FULL image is
                      added, with bbox and keypoints mirrored using the
                      full image width. This lets the detection component
                      learn to localize a hand (left or right orientation)
                      within a wider context (background/other structures),
                      unlike 'mirrored' which only ever sees tight crops.
    """
    def __init__(self, image_dir, sampled_ids, id_to_file, id_to_ann,
                 mode='full', transforms=None):
        self.image_dir  = image_dir
        self.transforms = transforms
        self.mode       = mode
        self.samples    = []  # list of (fname, ann, flip: bool)

        base_samples = match_samples(sampled_ids, id_to_file, id_to_ann)

        if mode in ('mirrored', 'mirrored_full'):
            for fname, ann in base_samples:
                self.samples.append((fname, ann, False))  # original left hand
                self.samples.append((fname, ann, True))   # mirrored synthetic right hand
        else:
            for fname, ann in base_samples:
                self.samples.append((fname, ann, False))

        print(f"Dataset: {len(self.samples)} valid samples (mode={mode})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, ann, flip = self.samples[idx]
        img = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")

        x, y, w, h = ann['bbox']
        kps       = ann['keypoints']
        keypoints = torch.tensor(kps, dtype=torch.float32).reshape(1, NUM_KEYPOINTS, 3)

        if self.mode in ('gt_crop', 'mirrored'):
            img = img.crop((x, y, x + w, y + h))
            boxes = torch.tensor([[0., 0., float(w), float(h)]], dtype=torch.float32)
            keypoints[0, :, 0] -= x
            keypoints[0, :, 1] -= y

            if flip:
                img = ImageOps.mirror(img)
                # Mirror x-coordinates within the crop width
                keypoints[0, :, 0] = float(w) - keypoints[0, :, 0]

        elif self.mode == 'mirrored_full':
            img_width, _img_height = img.size
            boxes = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)

            if flip:
                img = ImageOps.mirror(img)
                # Mirror x-coordinates using the FULL image width, not the crop width
                boxes[0, 0] = float(img_width) - (x + w)
                boxes[0, 2] = float(img_width) - x
                keypoints[0, :, 0] = float(img_width) - keypoints[0, :, 0]

        else:  # mode == 'full'
            boxes = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)

        labels = torch.ones((1,), dtype=torch.int64)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'gt_crop', 'mirrored', 'mirrored_full'],
                        help='full: whole image | gt_crop: crop with gt bbox | '
                             'mirrored: crop with gt bbox + mirrored synthetic right hand | '
                             'mirrored_full: whole image + mirrored whole image (right hand)')
    parser.add_argument('--full-dataset', action='store_true',
                        help='use the entire train/val split instead of the '
                             'stratified 250/50 subset')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Mode: {args.mode}")

    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    # Sampling
    if args.full_dataset:
        train_ids, id_to_file_train, id_to_ann_train = load_all_ids(
            f"{ANN}/RHPE_anatomical_ROIs_train.json",
            f"{ANN}/RHPE_Boneage_train.csv", "TRAIN")

        val_ids, id_to_file_val, id_to_ann_val = load_all_ids(
            f"{ANN}/RHPE_anatomical_ROIs_val.json",
            f"{ANN}/RHPE_Boneage_val.csv", "VAL")
    else:
        train_ids, id_to_file_train, id_to_ann_train = stratified_sample(
            f"{ANN}/RHPE_anatomical_ROIs_train.json",
            f"{ANN}/RHPE_Boneage_train.csv", 250, "TRAIN", seed=SEED)

        val_ids, id_to_file_val, id_to_ann_val = stratified_sample(
            f"{ANN}/RHPE_anatomical_ROIs_val.json",
            f"{ANN}/RHPE_Boneage_val.csv", 50, "VAL", seed=SEED)

    # Datasets and DataLoaders
    train_dataset = RHPEKeypointDataset(
        f"{BASE}/RHPE_train", train_ids, id_to_file_train, id_to_ann_train,
        mode=args.mode, transforms=transform)
    val_dataset = RHPEKeypointDataset(
        f"{BASE}/RHPE_val", val_ids, id_to_file_val, id_to_ann_val,
        mode=args.mode, transforms=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                              num_workers=4, pin_memory=True,
                              collate_fn=lambda x: tuple(zip(*x)))
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False,
                              num_workers=4, pin_memory=True,
                              collate_fn=lambda x: tuple(zip(*x)))

    # Model and optimizer — SGD with standard hyperparameters
    model     = get_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # 20 epochs as indicated in the BoNet paper for keypoint estimation
    num_epochs = 20

    suffix    = "_fulldata" if args.full_dataset else ""
    save_path = f"/data01/giacomelli/experiments/keypoints/checkpoints/keypointrcnn_rhpe_{args.mode}{suffix}.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_val_loss = float('inf')

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

        # Validation — loss (train-mode forward pass, no backward) as proxy
        # for best model selection: includes the keypoint loss term, unlike
        # bbox mAP. Matters for modes like gt_crop where the bbox target
        # never varies ([0,0,w,h]): mAP saturates early and stops
        # discriminating between epochs.
        model.train()
        val_loss_total = 0.0

        with torch.no_grad():
            for val_imgs, val_targets in val_loader:
                val_imgs_gpu    = [img.to(device) for img in val_imgs]
                val_targets_gpu = [{k: v.to(device) for k, v in t.items()} for t in val_targets]

                loss_dict = model(val_imgs_gpu, val_targets_gpu)
                val_loss_total += sum(loss_dict.values()).item()

        avg_val_loss = val_loss_total / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} ── "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  => New best model saved to {save_path} with Val Loss: {best_val_loss:.4f}")

    print(f"Training complete. Best Validation Loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()