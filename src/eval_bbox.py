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