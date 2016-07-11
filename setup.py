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

    @property
    def scalar(self):
        return self.__scalar

    @scalar.setter
    def scalar(self, scalar):
        self.__scalar = float(scalar)

    def __call__(self, y):
        return self.func(self.scalar*y)


input = {
    'Case name': 'CoFlow',
    'Show plot': True,
    'Contour levels': np.linspace(0, 1, 101),
    'nx': int(2001),
    'ny': int(201),
    'length_x': float(10),
    'length_y': float(1),
    'R': float(1),
    'Bi': [(2, 0.), (4, 1.), (10, 1.)],
    'NTU': float(1),
    'F(y)': BoundaryFunc(1.),
}

if __name__ == '__main__':
    bfunc = BoundaryFunc(np.square, int(1))

    print bfunc.scalar

    bfunc.__scalar = int(1)

    print bfunc.__scalar