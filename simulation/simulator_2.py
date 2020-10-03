import numpy as np
import sys
sys.path.append("/Users/anjianli/Desktop/robotics/project/optimized_dp")
sys.path.append("/home/anjianl/Desktop/project/optimized_dp")
from matplotlib import pyplot as plt
from matplotlib import animation, rc
from argparse import ArgumentParser

import pandas
import scipy.io
from scipy import integrate
import time

from simulation.optimal_control_reldyn5d import OptimalControlRelDyn5D
from prediction.process_prediction_v3 import ProcessPredictionV3
from prediction.predict_mode_v3 import PredictModeV3
from simulation.stanley_controller import *
from simulation.human_car import *
from simulation.optimal_control_bicycle4d import OptimalControlBicycle4D

class Simulator2(object):

    """
    For roundabout scenario simulation

    a big loop over time

    robot trajectory:
        - planning
        - optimal control take over
        - recover
    human car:
        - use prediction trajectory

    map

    plot function

    """

    def __init__(self):

        self.scenario = "roundabout"

        if '/Users/anjianli/anaconda3/envs/hcl-env/lib/python3.8' not in sys.path:
            self.file_dir_intersection = '/home/anjianl/Desktop/project/optimized_dp/data/intersection-data'
            self.file_dir_roundabout = '/home/anjianl/Desktop/project/optimized_dp/data/roundabout-data'
        else:
            # My laptop
            self.file_dir_intersection = '/Users/anjianli/Desktop/robotics/project/optimized_dp/data/intersection-data'
            self.file_dir_roundabout = '/Users/anjianli/Desktop/robotics/project/optimized_dp/data/roundabout-data'

        # Trial 1
        # self.huamn_car_file_name_roundabout = 'car_2.csv'
        # self.robot_car_file_name_roundabout = 'car_4_refPath.csv'
        # self.human_start_step = 0
        # self.robot_target_speed = 12
        # self.robot_start_step = 0

        # Trial 2
        self.huamn_car_file_name_roundabout = 'car_9.csv'
        self.robot_car_file_name_roundabout = 'car_6_refPath.csv'
        self.human_start_step = 5
        self.robot_target_speed = 2
        self.robot_start_step = 80

        # Trial 3
        # self.huamn_car_file_name_roundabout = 'car_24.csv'
        # self.robot_car_file_name_roundabout = 'car_6_refPath.csv'
        # self.human_start_step = 30
        # self.robot_target_speed = 2
        # self.robot_start_step = 45

        self.poly_num = 30

        self.show_animation = True
        # self.show_animation = False

        self.use_safe_control = True
        # self.use_safe_control = False

        self.use_prediction = True
        # self.use_prediction = False

        self.save_plot = True
        # self.save_plot = False

    def simulate(self):

        ################### PARSING ARGUMENTS FROM USERS #####################

        # parser = ArgumentParser()
        # parser.add_argument("-safe_controller", type=bool)
        # args = parser.parse_args()

        print("Start simulating")
        # print(args.safe_controller)

        # Set time
        tMax = 100
        dt = 0.1

        # Read prediction as human car's trajectory
        human_car_traj = self.get_traj_from_prediction(filename=self.huamn_car_file_name_roundabout)
        human_car_traj['x_t'] = [x + 1000 for x in human_car_traj['x_t']] # TODO: modify human car traj x
        human_car_traj['y_t'] = [y + 1000 for y in human_car_traj['y_t']] # TODO: modify human car traj y

        # Read ref path as robot_car_traj
        robot_car_traj_ref = self.get_traj_from_ref_path(filename=self.robot_car_file_name_roundabout)
        robot_car_traj_ref['x_t'] = [x + 1000 for x in robot_car_traj_ref['x_t']] # TODO: modify robot car traj x
        robot_car_traj_ref['y_t'] = [y + 1000 for y in robot_car_traj_ref['y_t']] # TODO: modify robot car traj y

        # Init human state for simulation
        human_start_step = self.human_start_step
        x_h_init, y_h_init, psi_h_init, v_h_init = self.init_human_state(human_car_traj, human_start_step)
        x_h_list, y_h_list, psi_h_list, v_h_list = [x_h_init], [y_h_init], [psi_h_init], [v_h_init]
        human_car = HumanState(x=x_h_init, y=y_h_init, psi=psi_h_init, v=v_h_init, ref_path=human_car_traj)

        # Init robot car object for simulation
        robot_start_step = self.robot_start_step
        target_speed = self.robot_target_speed
        x_r_init, y_r_init, psi_r_init, v_r_init = self.init_robot_state(robot_car_traj_ref, robot_start_step)
        v_r_init = self.robot_target_speed
        x_r_list, y_r_list, psi_r_list, v_r_list = [x_r_init], [y_r_init], [psi_r_init], [v_r_init]
        robot_car = RobotState(x=x_r_init, y=y_r_init, yaw=psi_r_init, v=v_r_init)

        # Init robot reference path
        ax = robot_car_traj_ref['x_t'][::10]
        ay = robot_car_traj_ref['y_t'][::10]
        cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=0.1)
        target_idx, _ = calc_target_index(robot_car, cx, cy)
        last_idx = len(cx) - 1

        #################################################################################
        # Main loop
        curr_t = 0
        curr_step_human = human_start_step
        min_dist = 100
        while curr_t < tMax and last_idx > target_idx and curr_step_human < len(human_car_traj['x_t']) - 2:

            try:
                # Get predicted trajectory of human car
                mode_num, mode_name, mode_probability = self.get_human_car_prediction(human_car_traj, curr_step_human)
            except IndexError:
                print("human car trajectory is finished")
                break

            ## TODO: In RelDyn5D, get reachability value function and optimal control for robot car
            rel_states, val_func_reldyn5d, optctrl_beta_r_reldyn5d, optctrl_a_r_reldyn5d, contour_rel_coordinate\
                = OptimalControlRelDyn5D(
                human_curr_states={'x_h': human_car.x_h, 'y_h': human_car.y_h, 'psi_h': human_car.psi_h, 'v_h': human_car.v_h},
                robot_curr_states={'x_r': robot_car.x, 'y_r': robot_car.y, 'psi_r': robot_car.yaw, 'v_r': robot_car.v},
                h_drv_mode=mode_num, h_drv_mode_pro=mode_probability, use_prediction=self.use_prediction).get_optctrl()

            ## TODO: In Bicycle4D, get reachability value function and optimal control for robot car
            _, val_func_bicycle4d, optctrl_beta_r_bicycle4d, optctrl_a_r_bicycle4d = OptimalControlBicycle4D(
                robot_curr_states={'x_r': robot_car.x, 'y_r': robot_car.y, 'psi_r': robot_car.yaw, 'v_r': robot_car.v},
                scenario=self.scenario).get_optctrl()

            # We get the contour coordinate, which is centered around robot car, and oriented at the robot car's heading
            # Thus when we transfer it to world coordinate, we should use (x * cos(yaw) - y * sin(yaw), x * sin(yaw) + y * cos(yaw))
            reachable_set_coordinate = np.asarray([
                contour_rel_coordinate[0, :] * np.cos(robot_car.yaw) - contour_rel_coordinate[1, :] * np.sin(
                    robot_car.yaw) + robot_car.x,
                contour_rel_coordinate[0, :] * np.sin(robot_car.yaw) + contour_rel_coordinate[1, :] * np.cos(
                    robot_car.yaw) + robot_car.y])

            print("Reldyn5D value function is", val_func_reldyn5d, ", Bicycle4D value function is", val_func_bicycle4d)

            curr_min_dist = np.sqrt(rel_states['x_rel'] ** 2 + rel_states['y_rel'] ** 2)
            min_dist = min(curr_min_dist, min_dist)
            # print("minimum distance is ", min_dist)

            if self.use_safe_control is True:
                # Check if reachability safe controller should be used
                if min(val_func_bicycle4d, val_func_reldyn5d) < 0:
                    if val_func_reldyn5d <= val_func_bicycle4d:
                        print("RelDyn5D safe controller in effect!")
                        print("optimal control beta is", optctrl_beta_r_reldyn5d, " acceleration is", optctrl_a_r_reldyn5d)
                        # time.sleep(3)

                        # TODO: don't use reachability controller, use deceleration
                        robot_car.safe_update(optctrl_beta_r_reldyn5d, optctrl_a_r_reldyn5d)
                        di, target_idx = stanley_control(robot_car, cx, cy, cyaw, target_idx)

                        # robot_car.update(-5, di)

                    else:
                        print("Bicycle4D safe controller in effect!")
                        print("optimal control beta is", optctrl_beta_r_bicycle4d, " acceleration is", optctrl_a_r_bicycle4d)
                        robot_car.safe_update(optctrl_beta_r_bicycle4d, optctrl_a_r_bicycle4d)
                else:
                    # If inside, reachable set, use optimal control to integrate the states
                    # Robot car
                    ai = pid_control(target_speed, robot_car.v)
                    di, target_idx = stanley_control(robot_car, cx, cy, cyaw, target_idx)
                    robot_car.update(ai, di)
            else:
                # If inside, reachable set, use optimal control to integrate the states
                # Robot car
                ai = pid_control(target_speed, robot_car.v)
                di, target_idx = stanley_control(robot_car, cx, cy, cyaw, target_idx)
                robot_car.update(ai, di)

            curr_t += dt
            curr_step_human += 1

            x_r_list.append(robot_car.x)
            y_r_list.append(robot_car.y)

            # Human car
            human_car.update(curr_step=curr_step_human)
            x_h_list.append(human_car.x_h)
            y_h_list.append(human_car.y_h)

            if '/Users/anjianli/anaconda3/envs/hcl-env/lib/python3.8' in sys.path:
                roundabout_curbs = np.load(
                    "/Users/anjianli/Desktop/robotics/project/optimized_dp/data/map/obstacle_map/roundabout_curbs.npy")
            else:
                roundabout_curbs = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/roundabout_curbs.npy")

            if self.show_animation:  # pragma: no cover
                plt.cla()
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(cx, cy, ".c", label="robot planning")
                plt.plot(x_r_list, y_r_list, "-b", label="robot trajectory")
                plt.plot(x_h_list, y_h_list, "-r", label="human trajectory")
                plt.plot(x_r_list[-1], y_r_list[-1], "xg", label="robot pos")
                plt.plot(x_h_list[-1], y_h_list[-1], "xr", label="human pos")
                plt.plot(reachable_set_coordinate[0, :], reachable_set_coordinate[1, :], "y")
                plt.scatter(roundabout_curbs[0], roundabout_curbs[1], color='black', linewidths=0.03)
                plt.axis("equal")
                plt.legend()
                plt.grid(True)
                if min(val_func_reldyn5d, val_func_bicycle4d) < 0 and self.use_safe_control is True:
                    if val_func_reldyn5d <= val_func_bicycle4d:
                        plt.title(
                            "robot speed (m/s):" + str(robot_car.v)[:4] + ", human mode:" + mode_name + ", RelDyn5D safe controller in effect!")
                    else:
                        plt.title(
                            "robot speed (m/s):" + str(robot_car.v)[
                                                   :4] + ", human mode:" + mode_name + ", Bicycle4D safe controller in effect!")
                else:
                    plt.title("robot speed (m/s):" + str(robot_car.v)[:4] + ", human mode:" + mode_name)
                plt.pause(0.1)

                if self.save_plot:
                    if '/Users/anjianli/anaconda3/envs/hcl-env/lib/python3.8' in sys.path:
                        plt.savefig("/Users/anjianli/Desktop/robotics/project/optimized_dp/result/simulation/1002/roundabout/2_np/t_{:.2f}_pred.png".format(curr_t))
                    else:
                        plt.savefig("/home/anjianl/Desktop/project/optimized_dp/result/simulation/1002/roundabout/2/t_{:.2f}_nopred.png".format(curr_t))

        print("minimum distance is ", min_dist)

        if self.show_animation:  # pragma: no cover
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, ".c", label="robot planning")
            plt.plot(x_r_list, y_r_list, "-b", label="robot trajectory")
            plt.plot(x_h_list, y_h_list, "-r", label="human trajectory")
            plt.plot(x_r_list[-1], y_r_list[-1], "xg", label="robot pos")
            plt.plot(x_h_list[-1], y_h_list[-1], "xr", label="human pos")
            plt.scatter(roundabout_curbs[0], roundabout_curbs[1], color='black', linewidths=0.03)
            plt.axis("equal")
            plt.legend()
            plt.grid(True)
            if min(val_func_reldyn5d, val_func_bicycle4d) < 0 and self.use_safe_control is True:
                if val_func_reldyn5d <= val_func_bicycle4d:
                    plt.title(
                        "robot speed (m/s):" + str(robot_car.v)[
                                               :4] + ", human mode:" + mode_name + ", RelDyn5D safe controller in effect!")
                else:
                    plt.title(
                        "robot speed (m/s):" + str(robot_car.v)[
                                               :4] + ", human mode:" + mode_name + ", Bicycle4D safe controller in effect!")
            else:
                plt.title("robot speed (m/s):" + str(robot_car.v)[:4] + ", human mode:" + mode_name)
            plt.show()

    def get_traj_from_prediction(self, filename):

        # Read trajectory from prediction
        if self.scenario == "roundabout":
            traj_file_name = self.file_dir_roundabout + '/' + filename
        else:
            traj_file_name = ""
        traj_file = pandas.read_csv(traj_file_name)
        length = len(traj_file)

        raw_traj = []
        traj_seg = {}
        traj_seg['x_t'] = []
        traj_seg['y_t'] = []
        traj_seg['v_t'] = []

        for i in range(length):
            traj_seg['x_t'].append(traj_file['x_t'][i])
            traj_seg['y_t'].append(traj_file['y_t'][i])
            traj_seg['v_t'].append(traj_file['v_t'][i])
            if traj_file['t_to_goal'][i] == 0:
                raw_traj.append(traj_seg)
                traj_seg = {}
                traj_seg['x_t'] = []
                traj_seg['y_t'] = []
                traj_seg['v_t'] = []
        # print(raw_traj)

        # raw_trajectory is cut into several piece. We can concatenate them together
        human_car_traj = {"x_t": [], "y_t": [], "v_t": []}
        for i in range(len(raw_traj)):
            human_car_traj["x_t"] += raw_traj[i]["x_t"]
            human_car_traj["y_t"] += raw_traj[i]["y_t"]
            human_car_traj["v_t"] += raw_traj[i]["v_t"]
        return human_car_traj

    def get_traj_from_ref_path(self, filename):

        # Read trajectory from prediction
        if self.scenario == "roundabout":
            traj_file_name = self.file_dir_roundabout + '/' + filename
        else:
            traj_file_name = ""

        traj_file = pandas.read_csv(traj_file_name, header=None, usecols=[0, 1], names=['x_t', 'y_t'],)
        length = len(traj_file)

        robot_car_traj = {}
        robot_car_traj['x_t'] = np.asarray(traj_file["x_t"]).tolist()
        robot_car_traj['y_t'] = np.asarray(traj_file["y_t"]).tolist()

        return robot_car_traj

    def init_robot_state(self, robot_car_traj_ref, robot_start_step):

        dx = (robot_car_traj_ref['x_t'][robot_start_step+1] - robot_car_traj_ref['x_t'][robot_start_step]) / 0.1
        dy = (robot_car_traj_ref['y_t'][robot_start_step+1] - robot_car_traj_ref['y_t'][robot_start_step]) / 0.1
        x_r_init, y_r_init = robot_car_traj_ref['x_t'][robot_start_step], robot_car_traj_ref['y_t'][robot_start_step]
        v_r_init = np.sqrt(dx ** 2 + dy ** 2)
        psi_r_init = np.arctan2(dy, dx)

        return x_r_init, y_r_init, psi_r_init, v_r_init

    def init_human_state(self, human_car_traj, human_start_time):

        dx = (human_car_traj['x_t'][human_start_time + 1] - human_car_traj['x_t'][human_start_time]) / 0.1
        dy = (human_car_traj['y_t'][human_start_time + 1] - human_car_traj['y_t'][human_start_time]) / 0.1
        x_h_init, y_h_init = human_car_traj['x_t'][human_start_time], human_car_traj['y_t'][human_start_time]
        v_h_init = np.sqrt(dx ** 2 + dy ** 2)
        psi_h_init = np.arctan2(dy, dx)

        return x_h_init, y_h_init, psi_h_init, v_h_init

    def get_human_car_prediction(self, human_car_traj, curr_step_human):

        traj_to_pred = {}
        if curr_step_human + self.poly_num < len(human_car_traj['x_t']) - 1:
            traj_to_pred['x_t'] = human_car_traj['x_t'][curr_step_human:curr_step_human + self.poly_num]
            traj_to_pred['y_t'] = human_car_traj['y_t'][curr_step_human:curr_step_human + self.poly_num]
            traj_to_pred['v_t'] = human_car_traj['v_t'][curr_step_human:curr_step_human + self.poly_num]
        else:
            traj_to_pred['x_t'] = human_car_traj['x_t'][curr_step_human:]
            traj_to_pred['y_t'] = human_car_traj['y_t'][curr_step_human:]
            traj_to_pred['v_t'] = human_car_traj['v_t'][curr_step_human:]

        # Fit polynomial for x, y position: x(t), y(t)
        poly_traj = ProcessPredictionV3().fit_polynomial_traj([traj_to_pred])

        raw_acc_list, raw_omega_list = ProcessPredictionV3().get_action_v_profile([traj_to_pred], poly_traj)

        filter_acc_list, filter_omega_list = PredictModeV3().filter_action(raw_acc_list, raw_omega_list)

        filter_acc, filter_omega = filter_acc_list[0], filter_omega_list[0]

        mode_num_list, mode_probability_list = PredictModeV3().get_mode(filter_acc, filter_omega)

        if mode_num_list[0] == 0:
            mode_name = "decelerate"
        elif mode_num_list[0] == 1:
            mode_name = "stable"
        elif mode_num_list[0] == 2:
            mode_name = "accelerate"
        elif mode_num_list[0] == 3:
            mode_name = "left turn"
        elif mode_num_list[0] == 4:
            mode_name = "right turn"
        elif mode_num_list[0] == 5:
            mode_name = "roundabout"
        else:
            mode_name = "other"

        return mode_num_list[0], mode_name, mode_probability_list[0]

if __name__ == "__main__":

    Simulator2().simulate()
