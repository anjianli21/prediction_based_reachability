import math
import numpy as np
import time
import sys

from prediction.process_prediction import ProcessPrediction
from prediction.predict_mode import PredictMode
from skimage import measure
from scipy.interpolate import RegularGridInterpolator

from matplotlib import pyplot as plt

class OptimalControlRelDyn5D(object):
    """"
    This module offers safety check for a pairwise car interaction (robot car, human car), and generates optimal control
    (beta_r, a_r) of collision avoidance for the robot car with 5D relative dynamics.

    Input:
    human current states: hcs = {x_h, y_h, v_h, psi_h}
    robot current states: rcs = {x_r, y_r, v_r, psi_r}
    human future states: hfs = {x_t, y_t}

    Computation:
    1. Derive relative dynamics: rels
    2. Obtain driving mode: drv_mode
    3. Switch to corresponding reachable set and optimal contro precomputation
    4. Interpolate value functio and control

    Output:
    optimal control: beta_r_opt, a_r_opt

    """

    def __init__(self, human_curr_states, robot_curr_states, h_drv_mode, h_drv_mode_pro, use_prediction, safe_data):

        # human_curr_states = {x_h, y_h, v_h, psi_h}
        self.human_curr_states = human_curr_states
        # robot_curr_states = {x_r, y_r, v_r, psi_r, beta_r}
        self.robot_curr_states = robot_curr_states
        # human_future_states = {x_h_t, y_h_t}
        if use_prediction:
            self.h_drv_mode = h_drv_mode
            self.h_drv_mode_pro = h_drv_mode_pro
        else:
            self.h_drv_mode = -1
            self.h_drv_mode_pro = [0, 0, 0, 0, 0, 0]

        # Data path
        # TODO: use latest brs 0929-correct
        # self.data_path = "/home/anjianl/Desktop/project/optimized_dp/data/brs/0929-correct/"
        # TODO; new target 1004-correct 4x3 radius
        # self.data_path = "/home/anjianl/Desktop/project/optimized_dp/data/brs/1004-correct/"
        # TODO: smaller new target 3.5x1.5 radius
        self.data_path = "/home/anjianl/Desktop/project/optimized_dp/data/brs/1006-correct/"

        # Computational bound for valfunc and optctrl
        # (x_rel, y_rel, psi_rel, v_h, v_r)
        self.x_rel_bound = [-15, 15]
        self.y_rel_bound = [-10, 10]
        self.psi_rel_bound = [-math.pi, math.pi]
        self.v_h_bound = [-1, 18] # New brs considers speed obstacles
        self.v_r_bound = [-1, 18] # New brs considers speed obstacles
        self.x_rel_dim = int(self.x_rel_bound[1] * 4 + 1)
        self.y_rel_dim = int(self.y_rel_bound[1] * 4 + 1)

        # Safe data is preloaded in simulation, so just use here
        # TODO: be careful, use self.h_drv_mode instead of h_drv_mode, because for no prediction, we set self.h_drv_mode = -1 manually
        self.valfunc = safe_data["reldyn5d"]["reldyn5d_brs_mode{:d}".format(self.h_drv_mode)]
        self.beta_r = safe_data["reldyn5d"]["reldyn5d_ctrl_beta_mode{:d}".format(self.h_drv_mode)]
        self.a_r = safe_data["reldyn5d"]["reldyn5d_ctrl_acc_mode{:d}".format(self.h_drv_mode)]

    def get_optctrl(self):

        # Get relative states
        self.rel_states = self.get_rel_states(self.human_curr_states, self.robot_curr_states)

        # print("relative states are", self.rel_states)

        if self.check_valid(self.rel_states):

            # # Choose valfunc and ctrl (beta_r, a_r)
            # self.valfunc_path = self.data_path + "mode{:d}/reldyn5d_brs_mode{:d}_t_3.00.npy".format(self.h_drv_mode, self.h_drv_mode)
            # self.beta_r_path = self.data_path + "mode{:d}/reldyn5d_ctrl_beta_mode{:d}_t_3.00.npy".format(self.h_drv_mode, self.h_drv_mode)
            # self.a_r_path = self.data_path + "mode{:d}/reldyn5d_ctrl_acc_mode{:d}_t_3.00.npy".format(self.h_drv_mode, self.h_drv_mode)
            #
            # self.valfunc = np.load(self.valfunc_path)
            # self.beta_r = np.load(self.beta_r_path)
            # self.a_r = np.load(self.a_r_path)

            # print("valfunc size", np.shape(self.valfunc))
            # print("beta_r size", np.shape(self.beta_r))
            # print("a_r size", np.shape(self.a_r))

            # Interpolation
            self.curr_valfunc = self.interpolate(self.valfunc, self.rel_states)
            self.curr_optctrl_beta_r = self.interpolate(self.beta_r, self.rel_states)
            self.curr_optctrl_a_r = self.interpolate(self.a_r, self.rel_states)
            end_time = time.time()

            # print("current value function is", self.curr_valfunc)
            # print("current beta_r", self.curr_optctrl_beta_r)
            # print("current a_r", self.curr_optctrl_a_r)

            # Get 0 level set
            try:
                self.contour_rel_coordinate = self.get_contour(self.valfunc, self.rel_states)
            except IndexError:
                self.contour_rel_coordinate = np.asarray([[0], [0]])

        else:
            # print("the relative states is outside of computation range!")
            self.curr_valfunc = 100
            self.curr_optctrl_beta_r = 0
            self.curr_optctrl_a_r = 0
            self.contour_rel_coordinate = np.asarray([[0], [0]])
            # print("current value function is", self.curr_valfunc)
            # print("current beta_r", self.curr_optctrl_beta_r)
            # print("current a_r", self.curr_optctrl_a_r)

        return self.rel_states, self.curr_valfunc, self.curr_optctrl_beta_r, self.curr_optctrl_a_r, self.contour_rel_coordinate

    def get_rel_states(self, human_curr_states, robot_curr_states):

        x_rel = np.cos(robot_curr_states['psi_r']) * (human_curr_states['x_h'] - robot_curr_states['x_r']) + \
                np.sin(robot_curr_states['psi_r']) * (human_curr_states['y_h'] - robot_curr_states['y_r'])
        y_rel = - np.sin(robot_curr_states['psi_r']) * (human_curr_states['x_h'] - robot_curr_states['x_r']) + \
                np.cos(robot_curr_states['psi_r']) * (human_curr_states['y_h'] - robot_curr_states['y_r'])
        psi_rel = human_curr_states['psi_h'] - robot_curr_states['psi_r']
        if psi_rel < -math.pi:
            psi_rel += math.pi * 2
        elif psi_rel > math.pi:
            psi_rel -= math.pi * 2
        v_h = human_curr_states['v_h']
        v_r = robot_curr_states['v_r']

        rel_states = {'x_rel': x_rel, 'y_rel': y_rel, 'psi_rel': psi_rel, 'v_h': v_h, 'v_r': v_r}

        return rel_states

    def get_drv_mode(self, human_future_states):

        length = len(human_future_states['x_t'])
        if length < 12:
            print("Warning! The length of human future states is less than requirement")

        poly_traj = ProcessPredictionV3().fit_polynomial_traj([human_future_states])
        acc, omega = ProcessPredictionV3().get_action_v_profile([human_future_states], poly_traj)

        mode_num, mode_probability = PredictModeV3().decide_mode(acc=acc[0], omega=omega[0])

        # print("current driving mode is", mode_num)
        # print("mode probability is", mode_probability)

        return mode_num, mode_probability

    def check_valid(self, rel_states):

        # (x_rel, y_rel, psi_rel, v_h, v_r)

        if (self.x_rel_bound[0] <= rel_states['x_rel'] <= self.x_rel_bound[1]) and \
                (self.y_rel_bound[0] <= rel_states['y_rel'] <= self.y_rel_bound[1]) and \
                (self.psi_rel_bound[0] <= rel_states['psi_rel'] <= self.psi_rel_bound[1]) and \
                (self.v_h_bound[0] <= rel_states['v_h'] <= self.v_h_bound[1]) and \
                (self.v_r_bound[0] <= rel_states['v_r'] <= self.v_r_bound[1]):
            return True
        else:
            return False

    def interpolate(self, data, point):

        x_rel = np.linspace(self.x_rel_bound[0], self.x_rel_bound[1], num=self.x_rel_dim)
        y_rel = np.linspace(self.y_rel_bound[0], self.y_rel_bound[1], num=self.y_rel_dim)
        psi_rel = np.linspace(-math.pi, math.pi, num=25)
        v_h = np.linspace(-1, 18, num=39)
        v_r = np.linspace(-1, 18, num=39)

        my_interpolating_function = RegularGridInterpolator((x_rel, y_rel, psi_rel, v_h, v_r), data)

        curr_x_rel = point['x_rel']
        curr_y_rel = point['y_rel']
        curr_psi_rel = point['psi_rel']
        curr_v_h = point['v_h']
        curr_v_r = point['v_r']

        curr_point = np.asarray([curr_x_rel, curr_y_rel, curr_psi_rel, curr_v_h, curr_v_r])

        return my_interpolating_function(curr_point)

    def get_contour(self, data, point):

        curr_v_h = point['v_h']
        curr_v_r = point['v_r']
        curr_psi_rel = point['psi_rel']

        v_h_index = int((curr_v_h - self.v_h_bound[0]) * 2)
        v_r_index = int((curr_v_r - self.v_r_bound[0]) * 2)
        psi_index = int((curr_psi_rel - self.psi_rel_bound[0]) / (math.pi / 12))

        x_y_val_func = np.squeeze(self.valfunc[:, :, psi_index, v_h_index, v_r_index])
        contour_set = np.squeeze(np.asarray(measure.find_contours(x_y_val_func, level=0)))
        # plt.plot(contour_set[:, 0], contour_set[:, 1])
        # plt.show()

        contour_rel_coordinate = np.asarray([contour_set[:, 0] * 0.5 + self.x_rel_bound[0], contour_set[:, 1] * 0.5 + self.y_rel_bound[0]])

        return contour_rel_coordinate

if __name__ == "__main__":

    hcs = {'x_h': 3, 'y_h': 5, 'v_h': 6, 'psi_h': - math.pi / 2}
    rcs = {'x_r': 3, 'y_r': 3, 'v_r': 0, 'psi_r': 0}
    hfs = {'x_t': [950.62, 951.22, 951.82, 952.42, 953.02, 953.62, 954.22, 954.83, 955.43, 956.03, 956.64, 957.24],
           'y_t': [986.06, 986.02, 985.99, 985.95, 985.91, 985.88, 985.84, 985.82, 985.78, 985.75, 985.7, 985.68],
           'v_t': [6.297153722000001, 6.301182825, 6.299214634, 6.29124924, 6.27629644, 6.254337775, 6.224393625,
                   6.187466767999999, 6.143547265, 6.092635555, 6.035747675, 5.971871147000001]}

    start_time = time.time()
    optctrl = OptimalControlRelDyn5D(human_curr_states=hcs, robot_curr_states=rcs, h_drv_mode=1, h_drv_mode_pro=[0, 1, 0, 0, 0, 0])
    optctrl.get_optctrl()

    end_time = time.time()
    print("overal all time is", end_time - start_time)