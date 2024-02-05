import numpy as np

def RandSubsetDataset(labels, frac, randStream=None):
    labels = np.array(labels).flatten()
    idx = []
    uniqLabels = np.unique(labels)
    for label in uniqLabels:
        catIndices = np.where(labels == label)[0]
        N = len(catIndices)
        if randStream is not None:
            np.random.set_state(randStream.get_state())
            randIdx = np.random.permutation(N)
        else:
            randIdx = np.random.permutation(N)
        randIdx = randIdx[:int(round(N*frac))]
        idx.extend(catIndices[randIdx])
    return idx
