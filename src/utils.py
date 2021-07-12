import numpy as np
import torch


# Function to seed everything for reproducibility
def seed_everything(seed, use_cuda=False):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if use_cuda:
        torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True