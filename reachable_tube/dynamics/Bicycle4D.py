import heterocl as hcl
import numpy as np
import math

from reachable_tube.helper.helper_math import my_atan
from reachable_tube.helper.helper_math import my_abs
from reachable_tube.helper.helper_math import my_min

class Bicycle_4D:
    """
    This class describe a 4D bicycle model (robot car)
    The dynamics is defined as follows:

    x_r' = v_r * cos(psi_r + beta_r)
    y_r' = v_r * sin(psi_r + beta_r)
    psi_r' = (v_r/l_r) * sin(beta_r)
    v_r' = a_r
    beta_r = tan^-1(l_r/(l_f + l_r) * tan(delta_f))

    Controls: beta_r, a_r
    Disturbances: None

    In code:
    State : state = (state[0], state[1], state[2], state[3]) = (x_r, y_r, psi_r, v_r)
    Control: uOpt = (uOpt[0], uOpt[1]) = (beta_r, a_r)
    Disturbance: dOpt = (dOpt[0], dOpt[1]) = None

    """
    def __init__(self, x=[0, 0, 0, 0, 0], uMin=np.array([-0.325, -5]), uMax=np.array([0.325, 3]), dims=4, uMode="max", dMode="min"):
        self.x = x
        self.uMode = uMode
        self.dMode = dMode

        # Object properties
        self.x = x

        # Control bounds
        self.uMin = uMin
        self.uMax = uMax

        # # Disturbance bounds
        # self.dMin = dMin
        # self.dMax = dMax

        self.dims = dims

        # Some constants
        self.l_r = 1.738
        self.l_f = 1.058

    def dynamics(self, t, state, uOpt, dOpt):
        """

        :param t:
        :param state:
        :param uOpt:
        :param dOpt:
        :return:
        """
        x1_dot = hcl.scalar(0, "x1_dot")
        x2_dot = hcl.scalar(0, "x2_dot")
        x3_dot = hcl.scalar(0, "x3_dot")
        x4_dot = hcl.scalar(0, "x4_dot")

        x1_dot[0] = state[3] * hcl.cos(state[2] + uOpt[0])
        x2_dot[0] = state[3] * hcl.sin(state[2] + uOpt[0])
        x3_dot[0] = (state[3] / self.l_r) * hcl.sin(uOpt[0])
        x4_dot[0] = uOpt[1]

        return (x1_dot[0], x2_dot[0], x3_dot[0], x4_dot[0])

    def opt_ctrl(self, t, state, spat_deriv):
        """
        For all the notation here, please refer to doc "reachability for relative dynamics"

        :param state:
        :param spat_deriv:
        :return:
        """

        # uOpt1: beta_r, uOpt2: a_r
        uOpt1 = hcl.scalar(0, "uOpt1")
        uOpt2 = hcl.scalar(0, "uOpt2")

        # # Define some constant
        c1 = hcl.scalar(0, "c1")
        c2 = hcl.scalar(0, "c2")

        # According to doc, c1, c2 are defined as follow
        c1[0] = - spat_deriv[0] * state[3] * hcl.sin(state[2]) + spat_deriv[1] * state[3] * hcl.cos(state[2]) + spat_deriv[2] * (state[3] / self.l_r)
        c2[0] = spat_deriv[0] * state[3] * hcl.cos(state[2]) + spat_deriv[1] * state[3] * hcl.sin(state[2])

        # Define some intermediate variables to store
        tmp1 = hcl.scalar(0, "tmp1")
        tmp2 = hcl.scalar(0, "tmp2")
        # Value these decision variable
        tmp1[0] = - my_atan(c2[0] / c1[0]) + math.pi / 2
        tmp2[0] = - my_atan(c2[0] / c1[0]) - math.pi / 2

        # Store umin and umax
        # uOpt = (uOpt[0], uOpt[1]) = (beta_r, a_r)
        umin1 = hcl.scalar(0, "umin1")
        umin2 = hcl.scalar(0, "umin2")
        umax1 = hcl.scalar(0, "umax1")
        umax2 = hcl.scalar(0, "umax2")
        umin1[0] = self.uMin[0]
        umin2[0] = self.uMin[1]
        umax1[0] = self.uMax[0]
        umax2[0] = self.uMax[1]

        # Just create and pass back, even though they're not used
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")

        with hcl.if_(self.uMode == "max"):

            # For uOpt1: beta_r
            # TODO: Some logic error fixed here
            with hcl.if_(c1[0] > 0):
                with hcl.if_(tmp1[0] >= umin1[0]):
                    with hcl.if_(tmp1[0] <= umax1[0]):
                        uOpt1[0] = tmp1[0]
                with hcl.if_(tmp1[0] > umax1[0]):
                    uOpt1[0] = umax1[0]
                with hcl.if_(tmp1[0] < umin1[0]):
                    uOpt1[0] = umin1[0]
            with hcl.if_(c1[0] < 0):
                with hcl.if_(tmp2[0] >= umin1[0]):
                    with hcl.if_(tmp2[0] <= umax1[0]):
                        uOpt1[0] = tmp2[0]
                with hcl.if_(tmp2[0] > umax1[0]):
                    uOpt1[0] = umax1[0]
                with hcl.if_(tmp2[0] < umin1[0]):
                    uOpt1[0] = umin1[0]
            with hcl.if_(c1[0] == 0):
                with hcl.if_(c2[0] >= 0):
                    with hcl.if_(0 >= umin1[0]):
                        with hcl.if_(0 <= umax1[0]):
                            uOpt1[0] = 0
                    with hcl.if_(0 < umin1[0]):
                        uOpt1[0] = my_min(my_abs(umin1[0]), my_abs(umax1[0]))
                    with hcl.if_(0 > umax1[0]):
                        uOpt1[0] = my_min(my_abs(umin1[0]), my_abs(umax1[0]))
                with hcl.if_(c2[0] < 0):
                    with hcl.if_(my_abs(umin1[0]) >= my_abs(umax1[0])):
                        uOpt1[0] = my_abs(umin1[0])
                    with hcl.if_(my_abs(umin1[0]) < my_abs(umax1[0])):
                        uOpt1[0] = my_abs(umax1[0])

            # For uOpt2: a_r
            with hcl.if_(spat_deriv[3] > 0):
                uOpt2[0] = umax2[0]
            with hcl.if_(spat_deriv[3] <= 0):
                uOpt2[0] = umin2[0]

        return (uOpt1[0], uOpt2[0], in3[0], in4[0])

    def optDstb(self, spat_deriv):
        """

        :param state:
        :param spat_deriv:
        :return:
        """

        # Just create and pass back, even though they're not used
        in1 = hcl.scalar(0, "in1")
        in2 = hcl.scalar(0, "in2")
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")

        return (in1[0], in2[0], in3[0], in4[0])