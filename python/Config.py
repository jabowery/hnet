import os

class Config:
    # path to output directory (relative to python working dir)
    OUT_DIR = os.path.join('..', '..', 'output_python')
    # path to dataset directory (relative to python working dir)
    DATASET_DIR = os.path.join('..', 'datasets')
    MIN_EDGES_PER_CMP = 4  # minimum number of edges per component
    DO_CACHE = False
    DO_INVERT_COLORS = True  # invert colors of nodes (true produces black text on white for mnist)
    DO_H_MODE = True  # if true, generate and use Hamiltonians in Energy.py; if false, convert to a boolean feed-forward network in Energy.py

    @staticmethod
    def MyDir():
        return os.path.dirname(os.path.abspath(__file__))
