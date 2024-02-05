from enum import Enum

class GRF(Enum):
    NULL = 0  # n/a, null
    GRID1D = 1  # 1d grid
    GRID2D = 2  # 2d rectangular grid w 1 channel
    GRID2DMULTICHAN = 3  # 2d rectangular grid w >1 channel
    FULL = 4  # fully connected
    SELF = 5  # each node is connected only to itself

