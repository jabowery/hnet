import numpy as np
from scipy.stats import norm

def encodeTransformScalar2SpatialScalar(data, n_spatial_stops, meta=None):
    assert np.min(data) >= 0 and np.max(data) <= 1, 'input scalar code must be in range 0 --> 1'
    data = np.squeeze(data)
    n_used_dims = np.sum(np.array(data.shape) > 1)
    if n_used_dims == 1:
        data = np.expand_dims(data, 0)  # place vector along dim 1

    d = [None] * n_spatial_stops
    for i in range(n_spatial_stops):
        mu = (i+1)/n_spatial_stops
        sd = mu / (2 * n_spatial_stops)
        d[i] = norm.pdf(data, mu, sd) / norm.pdf(mu, mu, sd)

    data = np.concatenate(d, axis=n_used_dims)

    spatialStop = np.zeros(data.shape)
    if n_used_dims >=1 and n_used_dims <= 5:
        indices = np.indices(data.shape)
        spatialStop = indices[n_used_dims]

    if meta:
        fn = list(meta.keys())
        r = [1] * n_used_dims + [n_spatial_stops]
        for i in range(len(fn)):
            if isinstance(meta[fn[i]], (np.ndarray, list)) and len(meta[fn[i]]) > 1:
                temp = np.squeeze(meta[fn[i]])
                if n_used_dims == 1:
                    temp = np.expand_dims(temp, 0)
                meta[fn[i]] = np.broadcast_to(temp, tuple(r))
    
    return data, spatialStop, meta
