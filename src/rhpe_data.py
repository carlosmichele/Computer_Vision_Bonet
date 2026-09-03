"""
Shared data-loading utilities for the RHPE dataset, used by both
train_bbox.py and train_keypoints.py.
"""
import os
import json
import pandas as pd


def load_coco_maps(json_path):
    """Parse a COCO-format annotation file.

    Returns (id_to_file, id_to_ann): COCO image id -> filename, and
    COCO image id -> its first annotation dict (bbox, keypoints, ...).
    """
    with open(json_path) as f:
        coco = json.load(f)

    id_to_file = {img['id']: img['file_name'] for img in coco['images']}
    id_to_ann  = {}
    for ann in coco['annotations']:
        if ann['image_id'] not in id_to_ann:
            id_to_ann[ann['image_id']] = ann

    return id_to_file, id_to_ann


def stratified_sample(json_path, csv_path, n_samples, split_name, seed=42):
    """Sample n_samples image IDs from csv_path, stratified by 24-month
    age bins, so the subset stays representative of the full age
    distribution instead of just taking the first N rows.
    """
    id_to_file, id_to_ann = load_coco_maps(json_path)

    df      = pd.read_csv(csv_path)
    age_col = 'Boneage' if 'Boneage' in df.columns else 'Chronological'
    df['bin'] = (df[age_col] // 24).astype(int)

    sampled_ids = []
    for bin_val, group in df.groupby('bin'):
        n       = max(1, round(n_samples * len(group) / len(df)))
        sampled = group.sample(min(n, len(group)), random_state=seed)
        sampled_ids.extend(sampled['ID'].tolist())

    sampled_ids = sampled_ids[:n_samples]
    print(f"{split_name}: {len(sampled_ids)} images sampled")
    return sampled_ids, id_to_file, id_to_ann


def load_all_ids(json_path, csv_path, split_name):
    """Same return shape as stratified_sample, but with no sampling —
    every image ID in csv_path is used.
    """
    id_to_file, id_to_ann = load_coco_maps(json_path)
    df  = pd.read_csv(csv_path)
    ids = df['ID'].tolist()
    print(f"{split_name}: {len(ids)} images (full dataset, no sampling)")
    return ids, id_to_file, id_to_ann


def match_samples(sampled_ids, id_to_file, id_to_ann):
    """Match a list of file-derived image IDs to their (filename,
    annotation) pairs. Builds the id_to_file reverse map once instead of
    scanning it per sampled ID (O(n+m) instead of O(n*m)).
    """
    file_id_to_coco_id = {
        int(os.path.splitext(fname)[0]): coco_id
        for coco_id, fname in id_to_file.items()
    }

    samples = []
    for img_id in sampled_ids:
        coco_id = file_id_to_coco_id.get(img_id)
        if coco_id is not None and coco_id in id_to_ann:
            samples.append((id_to_file[coco_id], id_to_ann[coco_id]))
    return samples
