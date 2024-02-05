import numpy as np
from scipy import sparse

def CountNumericOccurrences(x, uniqArr=None, dim=1):
    # Input validation
    x = np.array(x)
    assert np.issubdtype(x.dtype, np.number), 'input must be numeric'
    
    if uniqArr is None:
        uniqArr = np.unique(x[~np.isnan(x)])
    else:
        uniqArr = np.array(uniqArr)
        assert np.issubdtype(uniqArr.dtype, np.number), 'uniqArr must be numeric'
        assert np.all(uniqArr == np.unique(uniqArr)), 'uniqArr must be unique'
        assert np.all(np.diff(uniqArr) > 0), 'uniqArr must be sorted'

    n_uniq = len(uniqArr)

    # If x is empty
    if x.size == 0:
        return np.zeros_like(uniqArr, dtype=int)

    # If x is not a vector
    elif len(x.shape) > 1:
        if dim not in [1, 2]:
            raise ValueError('dim must be 1 or 2')
            
        counts_per_dim = []
        for xi in np.split(x, x.shape[dim-1], axis=dim-1):
            counts_per_dim.append(CountNumericOccurrences(xi, uniqArr=uniqArr))    
        
        return np.array(counts_per_dim)

    # Omit nan values
    x = x[~np.isnan(x)]

    # Optimized counting methods
    if n_uniq == np.max(uniqArr) and np.all(uniqArr == np.arange(1, n_uniq+1)):
        counts = np.bincount(x.astype(int), minlength=n_uniq+1)[1:]
    else:
        counts = np.array([np.sum(x == ua) for ua in uniqArr])

    if uniqArr.shape[0] == 1:
        return counts.reshape(-1,1)

    return counts
