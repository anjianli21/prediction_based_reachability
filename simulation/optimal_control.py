import math
import numpy as np

from prediction.process_prediction_v3 import ProcessPredictionV3
from prediction.predict_mode_v3 import PredictModeV3

class OptimalControl(object):
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

    def __init__(self, human_curr_states, robot_curr_states, human_future_states):

        # human_curr_states = {x_h, y_h, v_h, psi_h}
        self.human_curr_states = human_curr_states
        # robot_curr_states = {x_r, y_r, v_r, psi_r, beta_r}
        self.robot_curr_states = robot_curr_states
        # human_future_states = {x_h_t, y_h_t}
        self.human_future_states = human_future_states

        # Data path
        self.data_path = "/home/anjianl/Desktop/project/optimized_dp/data/brs/0910/"

    def get_optctrl(self):

        # Get relative states
        self.rel_states = self.get_rel_states(self.human_curr_states, self.robot_curr_states)

        # Get human's driving mode
        self.h_drv_mode, self.h_drv_mode_pro = self.get_drv_mode(self.human_future_states)

        # Choose valfunc and ctrl (beta_r, a_r)
        self.valfunc_path = self.data_path + "mode{:d}/reldyn5d_brs_mode{:d}_t_3.00.npy".format(self.h_drv_mode, self.h_drv_mode)
        self.beta_r_path = self.data_path + "mode{:d}/reldyn5d_ctrl_beta_mode{:d}_t_3.00.npy".format(self.h_drv_mode, self.h_drv_mode)
        self.a_r_path = self.data_path + "mode{:d}/reldyn5d_ctrl_acc_mode{:d}_t_3.00.npy".format(self.h_drv_mode, self.h_drv_mode)

        self.valfunc = np.load(self.valfunc_path)
        self.beta_r = np.load(self.beta_r_path)
        self.a_r = np.load(self.a_r_path)

        print("valfunc size", np.shape(self.valfunc))
        print("beta_r size", np.shape(self.beta_r))
        print("a_r size", np.shape(self.a_r))

        # Interpolation

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

        print("current driving mode is", mode_num)
        print("mode probability is", mode_probability)

        return mode_num, mode_probability

if __name__ == "__main__":

    hcs = {'x_h': 2, 'y_h': 3, 'v_h': 3, 'psi_h': math.pi / 6}
    rcs = {'x_r': 3, 'y_r': 5, 'v_r': 6, 'psi_r': math.pi / 4}
    hfs = {'x_t': [950.62, 951.22, 951.82, 952.42, 953.02, 953.62, 954.22, 954.83, 955.43, 956.03, 956.64, 957.24],
           'y_t': [986.06, 986.02, 985.99, 985.95, 985.91, 985.88, 985.84, 985.82, 985.78, 985.75, 985.7, 985.68],
           'v_t': [6.297153722000001, 6.301182825, 6.299214634, 6.29124924, 6.27629644, 6.254337775, 6.224393625,
                   6.187466767999999, 6.143547265, 6.092635555, 6.035747675, 5.971871147000001]}

    optctrl = OptimalControl(human_curr_states=hcs, robot_curr_states=rcs, human_future_states=hfs)
    optctrl.get_optctrl()