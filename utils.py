import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from autogluon.multimodal import MultiModalPredictor

def set_seed(seed: int):
    """
    Set random seeds for reproducibility across numpy, Python, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def path_expander(path: str, base_folder: str) -> str:
    """
    Expand relative paths in a semicolon-separated list to absolute paths.
    """
    parts = path.split(';')
    return ';'.join(
        os.path.abspath(os.path.join(base_folder, p))
        for p in parts
    )


def sample_batched(
    sampler,
    model,
    num_samples: int,
    X_test: np.ndarray,
    chunk_size: int = 50
) -> np.ndarray:
    """
    Generate `num_samples` per test point using batched sampling.
    Returns an array of shape [n_points, num_samples].
    """
    device = sampler.device
    condition = torch.tensor(X_test, dtype=torch.float32, device=device)
    n, dim = condition.shape
    all_chunks = []

    for _ in range(0, num_samples, chunk_size):
        csize = min(chunk_size, num_samples - len(all_chunks) * chunk_size)
        repeated = condition.unsqueeze(1).repeat(1, csize, 1).view(-1, dim)
        samples = sampler.sample(model, repeated.shape[0], repeated)
        samples = samples.detach().cpu().numpy().reshape(n, csize)
        all_chunks.append(samples)

    return np.concatenate(all_chunks, axis=1)


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )
    
def extract_embeddings(predictor: MultiModalPredictor, train_df, test_df,
                       label: str):
    """
    Extract multimodal embeddings using a fitted predictor.
    """
    train_emb = predictor.extract_embedding(
        train_df.drop(columns=[label]), as_pandas=False
    )
    test_emb = predictor.extract_embedding(
        test_df.drop(columns=[label]), as_pandas=False
    )
    return train_emb, test_emb