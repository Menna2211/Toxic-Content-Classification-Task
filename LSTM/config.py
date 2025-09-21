import numpy as np
import tensorflow as tf
import random
import os

# Configuration settings
SEED = 42
MAX_FEATURES = 1500
MAX_LENGTH = 75
BATCH_SIZE = 16
EPOCHS = 100
PATIENCE = 10

def set_seeds(seed=SEED):
    """Set seeds for reproducible results"""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'