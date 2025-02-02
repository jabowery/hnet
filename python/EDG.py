from enum import Enum

class EDG(Enum):
    NULL = 0  # n/a, null
    T = 1
    NOR = 2
    NCONV = 3
    NX = 4
    NIMPL = 5
    NY = 6
    XOR = 7
    NAND = 8
    AND = 9
    NXOR = 10
    Y = 11
    IMPL = 12
    X = 13
    CONV = 14
    OR = 15
    F = 16


    def Op(self):
        op = [
            [0, 0, 0, 0],   # T: F PREV T was reltype 0 so H as SpMat 0 accepts all
            [1, -1, 1, 0],  # NOR # H_NOR[0,0].2*H_NOR[0,1]+H_NOR[1,1],k_NOR
            [0, 1, -1, 1],  # NCONV
            [1, 0, 0, 0],   # NX
            [-1, 1, 0, 1],  # NIMPL
            [0, 0, 1, 0],   # NY
            [-1, 2, -1, 1], # XOR
            [0, 1, 0, 0],   # NAND
            [0, -1, 0, 1],  # AND
            [1, -2, 1, 0],  # NXOR
            [0, 0, -1, 1],  # Y
            [1, -1, 0, 0],  # IMPL
            [-1, 0, 0, 1],  # X
            [0, -1, 1, 0],  # CONV
            [-1, 1, -1, 1], # OR
            [0, 0, 0, 1]    # F: was T
        ]
        return op[self.value]

class EDGlist(list):

    def __init__(self, values):
        super(EDGlist,self).__init__([EDG(value) for value in values])

    def n(self):
        return len(self.values)
