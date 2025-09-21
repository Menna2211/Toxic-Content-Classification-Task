# config_transformer.py
import numpy as np
import torch
import random
import os

# Configuration settings
SEED = 42
BATCH_SIZE = 16
EPOCHS = 10
MAX_LENGTH = 256
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

def set_seeds(seed=SEED):
    """Set seeds for reproducible results across numpy, torch, and cuda"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device():
    """Check for GPU availability and return the device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device