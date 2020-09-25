import numpy as np

class Bicycle(object):
    """
    Bicycle dynamics for robot car

    """

    def __init__(self, x, y, psi, v):

        self.x = x

    def update(self):

        a = 0

