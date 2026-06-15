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