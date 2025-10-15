import os
import torch


class Config:

    MODEL_PATH = "/cephfs/shared/impact/LLM_models/llama3.1/Meta-Llama-3.1-8B-hf"
    OUTPUT_DIR = "/cephfs/shared/wjj/ICLR/LLAMAtest/dynamic_llama_output"

    TEACHER_DEVICE = torch.device("cuda:0")  # 默认设备
    STUDENT_DEVICE = torch.device("cuda:0")  # 默认设备

    HIDDEN_SIZE = 4096
    NUM_LAYERS = 32

    ADAPTER_BASE_SIZE = 256
    MAX_DISTANCE = 6
    MAX_BLOCKS = 6
    LAYER_EMBED_DIM = 64

    MIN_PATH_LENGTH = 28
    LENGTH_PENALTY_ALPHA = 0.05

    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    EPOCHS = 80
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0

    INIT_TEMP = 2
    MIN_TEMP = 0.8
    TEMP_DECAY = 0.95


    # Reproducibility settings
    SEED = 42

    # Evaluation frequency during training (0 to disable per-epoch eval)
    EVAL_EVERY = 1
    EVAL_TEMPERATURE = 0.1
    EVAL_ENFORCE_MIN_PATH = False


    # DISTANCE_SCHEDULE = {
    #     0: 2,
    #     10: 3,
    #     25: 4,
    #     40: 5
    # }

    DISTANCE_SCHEDULE = {
        0: 2,
        60: 3
    }

    DEBUG_MODE = False
    DEBUG_SAMPLE_RATIO = 1.0

    # Loss weights
    KD_WEIGHT = 1
    POLICY_WEIGHT = 1
    COG_WEIGHT = 0.25
    REP_WEIGHT = 1.0
    TASK_CE_WEIGHT = 2

    # TIDR parameters
    TIDR_BETA = 0.005

    # Tokenizer parameters
    MAX_LENGTH = 1024

    # Special tokens
    HCOG_TOKEN = "<hcog>"

    # FSDP parameters
    FSDP_SHARDING_STRATEGY = "FULL_SHARD"
    FSDP_MIN_PARAMS = 5000000
    FSDP_BACKWARD_PREFETCH = "BACKWARD_PRE"
    FSDP_CPU_OFFLOAD = False
    FSDP_STATE_DICT_TYPE = "FULL_STATE_DICT"  #

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    @staticmethod
    def get_max_skip_distance(epoch):

        max_distance = 1
        for e, dist in sorted(Config.DISTANCE_SCHEDULE.items()):
            if epoch >= e:
                max_distance = dist
        return max_distance