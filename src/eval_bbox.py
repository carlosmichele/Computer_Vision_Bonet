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