# /aster-deit/config.py

import torch

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Configuration ---
MODEL_ID = "/cephfs/shared/wjj/Deit"
TIMM_MODEL_NAME = 'deit_base_patch16_224'

# --- Dataset Configuration ---
TRAIN_DATA_PATH = "./data_splits/train.csv"
VAL_DATA_PATH = "./data_splits/val.csv"
TEST_DATA_PATH = "./data_splits/test.csv"
CLASSES_PY_PATH = '/cephfs/shared/wjj/nips/Vitlast/classes.py'


# --- Checkpoint and Logging Configuration ---
CHECKPOINT_DIR = "./checkpoints_deit"
EVAL_OUTPUT_DIR = "./evaluation_results_deit"

# --- Training Hyperparameters ---
# Recommended to increase CE_LOSS_WEIGHT to prioritize the main task
NUM_EPOCHS = 250
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
GRADIENT_ACCUMULATION_STEPS = 4
CLIP_GRAD_NORM = 1.0
LOG_INTERVAL = 10

# --- ASTER Component Dimensions ---
ADAPTER_BOTTLENECK_DIM = 128
SCORER_HIDDEN_DIM = 256

# --- Loss Composition Weights (Lambdas) ---
# It is highly recommended to increase the CE_LOSS_WEIGHT
CE_LOSS_WEIGHT = 10.0
RL_LOSS_WEIGHT = 0.5
KD_LOSS_WEIGHT = 1.0

# --- Reinforcement Learning (TIDR-V2) Hyperparameters ---
GAMMA = 0.99
W_EFFICIENCY = 0.01
W_TASK = 1.0
# --- NEW: Penalty for large skips to discourage overly aggressive behavior ---
SKIP_PENALTY_WEIGHT = 0.2

# --- Knowledge Distillation Hyperparameters ---
KD_TEMP = 1.0
