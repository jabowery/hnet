import numpy as np

def EqualizeN(label_idx, n=None):
    unique_labels, label_idx = np.unique(label_idx, return_inverse=True)
    if n is None or np.isnan(n):
        n = min(np.bincount(label_idx))
    else:
        assert n <= min(np.bincount(label_idx)), "n is larger than the smallest class size"
    
    selected_idx = np.zeros((n, len(unique_labels)), dtype=int)
    for i, label in enumerate(unique_labels):
        class_idx = np.where(label_idx == i)[0]
        selected_idx[:, i] = np.random.choice(class_idx, n, replace=False)
    
    selected_idx = np.sort(selected_idx.flatten())
    return selected_idx
