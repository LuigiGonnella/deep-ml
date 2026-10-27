import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    data: (n, d)
    """

    #shuffle data
    n = data.shape[0]
    shuffled_data = data[np.random.default_rng(seed).permutation(n)]
    
    #get split dims with floor division
    train_dim = int(n * train_frac)
    val_dim = int(n * validation_frac)
    
    #extract splits
    train_split = shuffled_data[: train_dim]
    val_split = shuffled_data[train_dim: train_dim + val_dim]
    test_split = shuffled_data[train_dim + val_dim:]

    return [train_split, val_split, test_split]
    