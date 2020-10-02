import math
import numpy as np
import time
import sys

from scipy.interpolate import RegularGridInterpolator

class OptimalControlBicycle4D(object):
    """"
    This module offers safety check for bicycle 4D w.r.t to road obstacles, and generates optimal control
    (beta_r, a_r) of collision avoidance.

    Input:
    robot current states: rcs = {x_r, y_r, v_r, psi_r}

    Computation:
    1. Interpolate value functio and control

    Output:
    optimal control: beta_r_opt, a_r_opt

    """

    def __init__(self, robot_curr_states, scenario):

        # robot_curr_states = {x_r, y_r, v_r, psi_r, beta_r}
        self.robot_curr_states = robot_curr_states
        self.scenario = scenario

        # Data path
        if self.scenario == "intersection":
            if '/Users/anjianli/anaconda3/envs/hcl-env/lib/python3.8' not in sys.path:
                # TODO: use latest brs 0929-correct
                self.data_path = "/home/anjianl/Desktop/project/optimized_dp/data/brs/0929-correct/obstacle/intersection/"
            else:
                self.data_path = "/Users/anjianli/Desktop/robotics/project/optimized_dp/data/brs/0929-correct/obstacle/intersection/"
        elif self.scenario == "roundabout":
            if '/Users/anjianli/anaconda3/envs/hcl-env/lib/python3.8' not in sys.path:
                # TODO: use latest brs 0929-correct
                self.data_path = "/home/anjianl/Desktop/project/optimized_dp/data/brs/0929-correct/obstacle/roundabout/"
            else:
                self.data_path = "/Users/anjianli/Desktop/robotics/project/optimized_dp/data/brs/0929-correct/obstacle/roundabout/"

        # Computational bound for valfunc and optctrl
        # (x_r, y_r, psi_r, v_r)
        if self.scenario == "intersection":
            self.x_r_bound = [940.8, 1066.7]
            self.y_r_bound = [935.8, 1035.1]
        elif self.scenario == "roundabout":
            self.x_r_bound = [956.7, 1073.4]
            self.y_r_bound = [954.0, 1046.0]
        self.psi_r_bound = [-math.pi, math.pi]
        self.v_r_bound = [-1, 18]

    def get_optctrl(self):

        print(self.robot_curr_states)

        if self.check_valid(self.robot_curr_states):

            # Choose valfunc and ctrl (beta_r, a_r)
            if self.scenario == "intersection":
                self.valfunc_path = self.data_path + "bicycle4d_brs_intersection_t_3.00.npy"
                self.beta_r_path = self.data_path + "bicycle4d_intersection_ctrl_beta_t_3.00.npy"
                self.a_r_path = self.data_path + "bicycle4d_intersection_ctrl_acc_t_3.00.npy"
            elif self.scenario == "roundabout":
                self.valfunc_path = self.data_path + "bicycle4d_brs_roundabout_t_3.00.npy"
                self.beta_r_path = self.data_path + "bicycle4d_roundabout_ctrl_beta_t_3.00.npy"
                self.a_r_path = self.data_path + "bicycle4d_roundabout_ctrl_acc_t_3.00.npy"

            self.valfunc = np.load(self.valfunc_path)
            self.beta_r = np.load(self.beta_r_path)
            self.a_r = np.load(self.a_r_path)

            # print("valfunc size", np.shape(self.valfunc))
            # print("beta_r size", np.shape(self.beta_r))
            # print("a_r size", np.shape(self.a_r))

            # Interpolation
            self.curr_valfunc = self.interpolate(self.valfunc, self.robot_curr_states)
            self.curr_optctrl_beta_r = self.interpolate(self.beta_r, self.robot_curr_states)
            self.curr_optctrl_a_r = self.interpolate(self.a_r, self.robot_curr_states)
            # print("current value function is", self.curr_valfunc)
            # print("current beta_r", self.curr_optctrl_beta_r)
            # print("current a_r", self.curr_optctrl_a_r)

        else:
            print("the robot states is outside of computation range!")
            self.curr_valfunc = 100
            self.curr_optctrl_beta_r = 0
            self.curr_optctrl_a_r = 0
            # print("current value function is", self.curr_valfunc)
            # print("current beta_r", self.curr_optctrl_beta_r)
            # print("current a_r", self.curr_optctrl_a_r)

        return self.robot_curr_states, self.curr_valfunc, self.curr_optctrl_beta_r, self.curr_optctrl_a_r

    def check_valid(self, robot_curr_states):

        # (x_rel, y_rel, psi_rel, v_h, v_r)

        if (self.x_r_bound[0] <= robot_curr_states['x_r'] <= self.x_r_bound[1]) and \
                (self.y_r_bound[0] <= robot_curr_states['y_r'] <= self.y_r_bound[1]) and \
                (self.psi_r_bound[0] <= robot_curr_states['psi_r'] <= self.psi_r_bound[1]) and \
                (self.v_r_bound[0] <= robot_curr_states['v_r'] <= self.v_r_bound[1]):
            return True
        else:
            return False

    def interpolate(self, data, point):

        if self.scenario == "intersection":
            x_r = np.linspace(940.8, 1066.7, num=466)
            y_r = np.linspace(935.8, 1035.1, num=366)
        elif self.scenario == "roundabout":
            x_r = np.linspace(956.7, 1073.4, num=465)
            y_r = np.linspace(954.0, 1046.0, num=367)
        psi_r = np.linspace(-math.pi, math.pi, num=25)
        v_r = np.linspace(-1, 18, num=39)

        my_interpolating_function = RegularGridInterpolator((x_r, y_r, psi_r, v_r), data)

        curr_x_r = point['x_r']
        curr_y_r = point['y_r']
        curr_psi_r = point['psi_r']
        curr_v_r = point['v_r']

        print(curr_x_r, curr_y_r, curr_psi_r, curr_v_r)

        curr_point = np.asarray([curr_x_r, curr_y_r, curr_psi_r, curr_v_r])

        return my_interpolating_function(curr_point)


if __name__ == "__main__":

    hcs = {'x_h': 3, 'y_h': 5, 'v_h': 6, 'psi_h': - math.pi / 2}
    rcs = {'x_r': 950.62, 'y_r': 986.06, 'v_r': 6.297153722000001, 'psi_r': 0}
    hfs = {'x_t': [950.62, 951.22, 951.82, 952.42, 953.02, 953.62, 954.22, 954.83, 955.43, 956.03, 956.64, 957.24],
           'y_t': [986.06, 986.02, 985.99, 985.95, 985.91, 985.88, 985.84, 985.82, 985.78, 985.75, 985.7, 985.68],
           'v_t': [6.297153722000001, 6.301182825, 6.299214634, 6.29124924, 6.27629644, 6.254337775, 6.224393625,
                   6.187466767999999, 6.143547265, 6.092635555, 6.035747675, 5.971871147000001]}

    start_time = time.time()
    optctrl = OptimalControlBicycle4D(robot_curr_states=rcs)
    print(optctrl.get_optctrl())

    end_time = time.time()
    print("overal all time is", end_time - start_time)