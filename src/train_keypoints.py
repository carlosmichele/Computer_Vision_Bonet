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

class RHPEKeypointDataset(Dataset):
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

    def __getitem__(self, idx):
        fname, ann = self.samples[idx]
        img = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")

        x, y, w, h = ann['bbox']
        boxes  = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)
        labels = torch.ones((1,), dtype=torch.int64)

        kps       = ann['keypoints']
        keypoints = torch.tensor(kps, dtype=torch.float32).reshape(1, NUM_KEYPOINTS, 3)

        target = {
            'boxes':     boxes,
            'labels':    labels,
            'keypoints': keypoints,  
            'image_id':  torch.tensor([idx])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target
    
    def get_model(num_keypoints=NUM_KEYPOINTS):
        model       = keypointrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
        model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_features, num_keypoints)
        return model
    
    