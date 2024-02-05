import numpy as np

def encodeTransformScalar2SpikeViaKWTA(data, n_winners, dim, min2Win=0):
    assert n_winners > 0, "n_winners must be positive"
    assert np.amin(data) >= 0 and np.amax(data) <= 1, 'input scalar code must be in range 0-->1'
    assert dim < len(data.shape), "dim must be lower than data dimensions"

    if min2Win > 0:
        isAboveMin = data > min2Win
        
    if n_winners == 1:
        idx = np.argmax(data, axis=dim) # idx contains indices into data
    else:
        idx = np.zeros(n_winners, dtype=int)
        for i in range(n_winners):
            max_ind = np.argmax(data, axis=dim)
            idx[i] = np.ravel_multi_index(max_ind, data.shape)
            if i < (n_winners - 1): # If we're not on the last winner
                data[np.unravel_index(idx[i], data.shape)] = -np.inf
    transformed_data = np.zeros_like(data, dtype=bool)
    transformed_data[np.unravel_index(idx, transformed_data.shape)] = True

    if min2Win > 0:
        transformed_data = transformed_data & isAboveMin
    return transformed_data
