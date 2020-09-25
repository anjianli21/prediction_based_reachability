import numpy as np

class HumanState(object):
    """
    Class representing the state of a vehicle.
    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """

    def __init__(self, x=0.0, y=0.0, psi=0.0, v=0.0, ref_path=None):
        """Instantiate the object."""
        self.x_h = x
        self.y_h = y
        self.psi_h = psi
        self.v_h = v

        self.ref_path = ref_path

    def update(self, curr_step):
        """
        Because we directly use the
        """
        next_step = curr_step + 1

        dx = (self.ref_path['x_t'][next_step + 1] - self.ref_path['x_t'][next_step]) / 0.1
        dy = (self.ref_path['y_t'][next_step + 1] - self.ref_path['y_t'][next_step]) / 0.1
        self.x_h, self.y_h = self.ref_path['x_t'][next_step], self.ref_path['y_t'][next_step]
        self.v_h = np.sqrt(dx ** 2 + dy ** 2)
        self.psi_h = np.arctan2(dy, dx)
