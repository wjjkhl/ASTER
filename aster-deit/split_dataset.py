# /aster-deit/split_dataset.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import sys

VAL_DIR = '/cephfs/shared/wjj/nips/imagenet/val'
CLASSES_PY_PATH = '/cephfs/shared/wjj/nips/Vitlast/classes.py'
OUTPUT_DIR = './data_splits'  

def load_module_from_file(file_path, module_name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise FileNotFoundError(f"Could not find the module file at {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def create_dataset_splits():
    print("--- Starting ImageNet Split Creation ---")

    try:
        classes_module = load_module_from_file(CLASSES_PY_PATH, "imagenet_classes")
        # IMAGENET2012_CLASSES is a dict like {'n01440764': ['tench', 'Tinca tinca'], ...}
        synset_to_class_idx = {synset: i for i, synset in enumerate(classes_module.IMAGENET2012_CLASSES.keys())}
        print(f"Successfully loaded {len(synset_to_class_idx)} class mappings.")
    except Exception as e:
        print(f"[FATAL ERROR] Could not load or parse the classes.py file. Error: {e}")
        return

    image_files = sorted([f for f in os.listdir(VAL_DIR) if f.lower().endswith('.jpeg')])
    print(f"Found {len(image_files)} total images in {VAL_DIR}")

    samples = []
    for filename in image_files:
        try:
            # Extract synset from filename like 'ILSVRC2012_val_00000001_n01751748.JPEG'
            synset_id = filename.split('.')[0].split('_')[-1]
            if synset_id in synset_to_class_idx:
                class_idx = synset_to_class_idx[synset_id]
                samples.append({'filepath': os.path.join(VAL_DIR, filename), 'label': class_idx})
        except (IndexError, ValueError):
            print(f"Warning: Could not parse synset ID from filename: {filename}")

    if not samples:
        print("[FATAL ERROR] No valid samples found. Check directory path and file naming convention.")
        return

    df = pd.DataFrame(samples)
    print(f"Created initial DataFrame with {len(df)} samples.")

    # 80% train, 20% temp (for val/test)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    # Split temp 50/50 to get 10% val, 10% test
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    print(f"\nSplit sizes:")
    print(f"  Training set:   {len(train_df)} samples")
    print(f"  Validation set: {len(val_df)} samples")
    print(f"  Test set:       {len(test_df)} samples")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_path = os.path.join(OUTPUT_DIR, 'train.csv')
    val_path = os.path.join(OUTPUT_DIR, 'val.csv')
    test_path = os.path.join(OUTPUT_DIR, 'test.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSuccessfully saved splits to:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"  - {test_path}")
    print("--- Split Creation Complete ---")


if __name__ == '__main__':

    create_dataset_splits()
