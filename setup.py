"""
Author: Adam O'Brien
"""

import numpy as np
import math
import collections

class BoundaryFunc(object):
    def __init__(self, func, scalar = 1.):

        if isinstance(func, float) or isinstance(func, int):
            self.func = lambda y: np.array([func]*len(y))
        else:
            self.func = func

        self.scalar = scalar

    def __call__(self, y):
        return self.func(self.scalar*y)


input = {
    'nx': int(1900),
    'ny': int(190),
    'length_x': float(10),
    'length_y': float(1),
    'R': float(1),
    'Bi': float(1),
    'NTU': float(1),
    'F(y)': BoundaryFunc(np.sin, math.pi),
}

if __name__ == '__main__':
    bfunc = BoundaryFunc(np.square, 1.)

    print bfunc(np.linspace(0, 1, input['ny']))