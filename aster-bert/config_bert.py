# /aster-bert/config_bert.py

import torch

SEED = 42
# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Configuration ---
# [MODIFIED] Changed model ID to the specified DistilBERT model fine-tuned on SST-2.
MODEL_ID = 'distilbert-base-uncased-finetuned-sst-2-english'

# --- Dataset Configuration ---
# This already correctly points to the SST-2 dataset.
DATASET_NAME = "glue"
DATASET_CONFIG = "sst2"
DATASET_TEXT_COLUMN = "sentence"


# --- Checkpoint and Logging Configuration ---
# [MODIFIED] Changed directory to reflect the new model.
CHECKPOINT_DIR = "./checkpoints_distilbert_sst2"
EVAL_OUTPUT_DIR = "./evaluation_results_distilbert_sst2"

# --- Training Hyperparameters ---
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
GRADIENT_ACCUMULATION_STEPS = 4
CLIP_GRAD_NORM = 1.0
LOG_INTERVAL = 10
MAX_SEQ_LENGTH = 128

# --- ASTER Component Dimensions ---
ADAPTER_BOTTLENECK_DIM = 128
SCORER_HIDDEN_DIM = 256

# --- Loss Composition Weights (Lambdas) ---
CE_LOSS_WEIGHT = 10
RL_LOSS_WEIGHT = 0.5
KD_LOSS_WEIGHT = 1.0

# --- Reinforcement Learning (TIDR-V2) Hyperparameters ---
GAMMA = 0.99
W_EFFICIENCY = 0.01
W_TASK = 1.0
SKIP_PENALTY_WEIGHT = 0.1

# --- Knowledge Distillation Hyperparameters ---
KD_TEMP = 1.0

MIN_TOTAL_EXECUTED_LAYERS = 2