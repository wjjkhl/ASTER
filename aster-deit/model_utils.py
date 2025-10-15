# /aster-deit/model_utils.py

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import timm
from timm.data import create_transform

import config


def load_model_and_processor():
    """
    Loads the DeiT model and its corresponding image processor/transform.
    The transform is created using `timm` to match the original DeiT training recipe.
    """
    print(f"--- Loading Model and Processor for {config.MODEL_ID} ---")

    device = config.DEVICE

    # 1. Load Model from local path
    print(f"Loading model from local path: {config.MODEL_ID}")
    model = AutoModelForImageClassification.from_pretrained(
        config.MODEL_ID,
        output_hidden_states=True  # Crucial for ASTER
    ).to(device)
    model.eval()  # Base model is frozen and used for feature extraction

    # 2. Create Image Transform using `timm`
    # This ensures the preprocessing matches the model's original training
    print(f"Creating data transforms using 'timm' recipe for '{config.TIMM_MODEL_NAME}'")
    timm_model_for_cfg = timm.create_model(config.TIMM_MODEL_NAME, pretrained=False)
    timm_config = timm_model_for_cfg.default_cfg

    transform = create_transform(
        input_size=timm_config['input_size'],
        is_training=False,  # We are doing inference-style training of adapters
        interpolation=timm_config['interpolation'],
        mean=timm_config['mean'],
        std=timm_config['std'],
        crop_pct=timm_config['crop_pct'],
    )
    print("Model and image transform loaded successfully.")

    return model, transform