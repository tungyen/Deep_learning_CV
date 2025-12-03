import torch
import numpy as np

def random_dropout(indices, max_dropout=0.95):
    dropout = np.random.random() * max_dropout
    drop_idx = np.where(np.random.random(len(indices)) < dropout)[0]
    if len(drop_idx) > 0:
        indices[drop_idx] = indices[0]
    return indices