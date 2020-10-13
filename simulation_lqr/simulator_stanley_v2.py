import sys
sys.path.append("/home/anjianl/Desktop/project/optimized_dp/")
import pandas
import os
import shutil
import json
import time
import pickle
import math
from PIL import Image

from simulation_lqr.stanley_control import *
from simulation_lqr.simulator_lqr_helper import SimulatorLQRHelper
from simulation_lqr.human_car import HumanState
from simulation_lqr.optimal_control_reldyn5d import OptimalControlRelDyn5D
from simulation_lqr.optimal_control_bicycle4d import OptimalControlBicycle4D

class SimulatorStanleyV2(SimulatorLQRHelper):

    def __init__(self):
        super(SimulatorStanleyV2, self).__init__()

        # Update frequency for the prediction, dt = 0.1s
        self.mode_predict_span = 1
        # When making prediction, choose trajectory over some episode
        self.episode_len = 12

        self.intersection_curbs = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/intersection_curbs.npy")
        self.roundabout_curbs = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/roundabout_curbs.npy")

    def simulate(self):

        print("Start Stanley simulating")

        ########################### Prepare Trajectory ########################################################################
        # Configure path
        if self.scenario == "intersection":
            human_car_file_name = self.huamn_car_file_name_intersection
            robot_car_file_name = self.robot_car_file_name_intersection
        elif self.scenario == "roundabout":
            human_car_file_name = self.huamn_car_file_name_roundabout
            robot_car_file_name = self.robot_car_file_name_roundabout

        # Read prediction as human car's trajectory
        human_car_traj = self.get_traj_from_prediction(scenario=self.scenario, filename=human_car_file_name)
        # Init human state for simulation
        x_h_init, y_h_init, psi_h_init, v_h_init = human_car_traj['x_t'][self.human_start_step], \
                                                   human_car_traj['y_t'][self.human_start_step], \
                                                   human_car_traj['psi_t'][self.human_start_step], \
                                                   human_car_traj['v_t'][self.human_start_step]
        HumanCar = HumanState(x=x_h_init, y=y_h_init, psi=psi_h_init, v=v_h_init, ref_path=human_car_traj)

        # Read ref path as robot_car_traj
        robot_car_traj_ref = self.get_traj_from_ref_path(scenario=self.scenario, filename=robot_car_file_name)
        x_r_init, y_r_init, psi_r_init, v_r_init = self.init_robot_state(robot_car_traj_ref, self.robot_start_step)
        v_r_init = self.robot_target_speed
        RobotCar = RobotState(x=x_r_init, y=y_r_init, yaw=psi_r_init, v=v_r_init)

        # Init robot reference path
        ax = robot_car_traj_ref['x_t'][::10]
        ay = robot_car_traj_ref['y_t'][::10]
        robot_goal = [ax[-1], ay[-1]]
        cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=0.1)
        target_idx, _, _ = calc_target_index(RobotCar, cx, cy)
        last_idx = len(cx) - 1

        # Initialize trajectory list for robot car and human car
        x_h_list, y_h_list, psi_h_list, v_h_list = [HumanCar.x_h], [HumanCar.y_h], [HumanCar.psi_h], [HumanCar.v_h]
        x_r_list, y_r_list, psi_r_list, v_r_list = [RobotCar.x], [RobotCar.y], [RobotCar.yaw], [RobotCar.v]
        t_list = [0]

        ########################### Main loop ########################################################################
        max_t = self.max_t  # max simulation time
        goal_dis = 0.3
        robot_stop_speed = self.robot_target_speed

        e, e_th = 0.0, 0.0

        # Variable initialization
        curr_t = 0.0
        curr_step_human = self.human_start_step
        curr_step = 0
        min_dist = 100
        max_deviation = 0

        time_use_reldyn5d_control = 0
        time_use_bicycl4d_control = 0

        # Main loop
        while curr_t <= max_t:

            start_time = time.time()

            # Get prediction ############################################################################################################
            if int(curr_step) % self.mode_predict_span == 0 and curr_step_human <= len(human_car_traj["x_t"]) - 1:
                try:
                    mode_num, mode_name, mode_probability = self.get_human_car_prediction(human_car_traj,
                                                                                          curr_step_human,
                                                                                          episode_len=self.episode_len)
                except IndexError:
                    print("human car trajectory is finished")

            # Get safe control ###########################################################################################################
            rel_states, val_func_reldyn5d, optctrl_beta_r_reldyn5d, optctrl_a_r_reldyn5d, contour_rel_coordinate \
                = OptimalControlRelDyn5D(
                human_curr_states={'x_h': HumanCar.x_h, 'y_h': HumanCar.y_h, 'psi_h': HumanCar.psi_h,
                                   'v_h': HumanCar.v_h},
                robot_curr_states={'x_r': RobotCar.x, 'y_r': RobotCar.y, 'psi_r': RobotCar.yaw, 'v_r': RobotCar.v},
                h_drv_mode=mode_num, h_drv_mode_pro=mode_probability, use_prediction=self.use_prediction,
                safe_data=self.safe_data).get_optctrl()

            _, val_func_bicycle4d, optctrl_beta_r_bicycle4d, optctrl_a_r_bicycle4d = OptimalControlBicycle4D(
                robot_curr_states={'x_r': RobotCar.x, 'y_r': RobotCar.y, 'psi_r': RobotCar.yaw, 'v_r': RobotCar.v},
                scenario=self.scenario,
                safe_data=self.safe_data).get_optctrl()

            reachable_set_coordinate = np.asarray([
                contour_rel_coordinate[0, :] * np.cos(RobotCar.yaw) - contour_rel_coordinate[1, :] * np.sin(
                    RobotCar.yaw) + RobotCar.x,
                contour_rel_coordinate[0, :] * np.sin(RobotCar.yaw) + contour_rel_coordinate[1, :] * np.cos(
                    RobotCar.yaw) + RobotCar.y])

            # Get Stanley controller for robot car ##########################################################################################
            di, target_idx, curr_max_deviation = stanley_control(RobotCar, cx, cy, cyaw, target_idx)
            ai = pid_control(self.robot_target_speed, RobotCar.v)

            # Update Human Car ###########################################################################################################
            if curr_step_human <= len(human_car_traj["x_t"]) - 2:
                HumanCar.update(curr_step=curr_step_human)
            x_h_list.append(HumanCar.x_h)
            y_h_list.append(HumanCar.y_h)
            psi_h_list.append(HumanCar.psi_h)
            v_h_list.append(HumanCar.v_h)

            # Update Robot Car based on safe control ###################################################################################
            if self.use_safe_control and min(val_func_bicycle4d, val_func_reldyn5d) < 0:
                if val_func_reldyn5d <= val_func_bicycle4d:
                    print("Use reldyn5d controller")
                    RobotCar.safe_update(optctrl_a_r_reldyn5d, optctrl_beta_r_reldyn5d)
                    time_use_reldyn5d_control += 1
                else:
                    RobotCar.safe_update(optctrl_a_r_bicycle4d, optctrl_beta_r_bicycle4d)
                    print("Use bicycle4d controller")
                    time_use_bicycl4d_control += 1
            else:
                RobotCar.update(ai, di) # Use LQR controller

            # Append the current states to the list
            x_r_list.append(RobotCar.x)
            y_r_list.append(RobotCar.y)
            psi_r_list.append(RobotCar.yaw)
            v_r_list.append(RobotCar.v)
            t_list.append(curr_t)

            # Update time ###########################################################################################################
            curr_t += dt
            curr_step += 1
            curr_step_human += 1

            # Statistics ###########################################################################################################
            max_deviation = max(max_deviation, abs(curr_max_deviation))
            # print("maximum deviation is", max_deviation)
            curr_min_dist = np.sqrt(rel_states['x_rel'] ** 2 + rel_states['y_rel'] ** 2)
            min_dist = min(curr_min_dist, min_dist)
            # print("minimum car distance is", min_dist)

            # check goal ###########################################################################################################
            dx = RobotCar.x - robot_goal[0]
            dy = RobotCar.y - robot_goal[1]
            if math.hypot(dx, dy) <= goal_dis:
                print("Goal")
                break

            # Plot
            if self.show_animation:
                plt.cla()
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(cx, cy, ".c", label="course")
                plt.plot(x_r_list, y_r_list, "-b", label="robot trajectory")
                plt.plot(x_h_list, y_h_list, "-r", label="human trajectory")
                plt.plot(x_r_list[-1], y_r_list[-1], "xg", label="robot pos")
                plt.plot(x_h_list[-1], y_h_list[-1], "xr", label="human pos")
                if self.use_safe_control:
                    plt.plot(reachable_set_coordinate[0, :], reachable_set_coordinate[1, :], "y")
                # plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
                if self.scenario == "intersection":
                    plt.scatter(self.intersection_curbs[0], self.intersection_curbs[1], color='black', linewidths=0.03)
                elif self.scenario == "roundabout":
                    plt.scatter(self.roundabout_curbs[0], self.roundabout_curbs[1], color='black', linewidths=0.03)
                plt.axis("equal")
                plt.grid(True)
                plt.title("v_r (m/s):" + str(round(RobotCar.v, 2)) + ", devia:" + str(round(max_deviation, 2)) +
                          ", dist:" + str(round(min_dist, 2)) + ", use pred:" + str(self.use_prediction) + ", mode:" + str(mode_name))
                plt.pause(0.001)

            if self.save_plot:
                tmp_folder_path = self.fig_save_dir + "tmp/"
                if not os.path.exists(tmp_folder_path):
                    os.mkdir(tmp_folder_path)
                plt.savefig(tmp_folder_path + "t_{:.2f}.png".format(curr_t))

            end_time = time.time()
            print("this step takes", end_time - start_time, "s")

        # Save statistics ###########################################################################################################
        print("this episode finishes!")
        self.max_deviation_list.append(max_deviation)
        self.min_dist_list.append(min_dist)
        self.time_use_reldyn5d_control.append(time_use_reldyn5d_control)
        self.time_use_bicycl4d_control.append(time_use_bicycl4d_control)
        self.final_target_index_list.append(target_idx)
        self.final_deviation_list.append(curr_max_deviation)

        # Convert png to gif ###########################################################################################################
        if self.save_plot:
            # Create the frames
            frames = []
            # imgs = glob.glob("*.png")
            imgs = []
            for i in np.arange(0.10, curr_t, 0.10):
                imgs.append(
                    tmp_folder_path + "t_{:.2f}.png".format(i))
            for i in imgs:
                new_frame = Image.open(i)
                frames.append(new_frame)

            # Save into a GIF file that loops forever
            save_dir = self.fig_save_dir
            if self.use_safe_control:
                if self.use_prediction:
                    file_name = self.figure_file_name + "_with_prediction_safe_control.gif"
                else:
                    file_name = self.figure_file_name + "_no_prediction_safe_control.gif"
            else:
                file_name = self.figure_file_name + "_no_safe_control.gif"
            frames[0].save(
                save_dir + file_name,
                format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=300, loop=0)

            # Delete tmp dir
            shutil.rmtree(tmp_folder_path)

        return 0

    def load_safe_data(self):

        self.safe_data = {}

        # load Bicycle4D data ####################################################################################################
        # Data path
        if self.scenario == "intersection":
            # TODO: use buffer area 1m
            self.data_path_bicycle4d = "/home/anjianl/Desktop/project/optimized_dp/data/brs/1006-correct/obstacle_buffer_1m/intersection/"
        elif self.scenario == "roundabout":
            # TODO: use buffer area 1m
            self.data_path_bicycle4d = "/home/anjianl/Desktop/project/optimized_dp/data/brs/1006-correct/obstacle_buffer_1m/roundabout/"

        if self.scenario == "intersection":
            self.valfunc_path_bicycle4d = self.data_path_bicycle4d + "bicycle4d_brs_intersection_t_3.00.npy"
            self.beta_r_path_bicycle4d = self.data_path_bicycle4d + "bicycle4d_intersection_ctrl_beta_t_3.00.npy"
            self.a_r_path_bicycle4d = self.data_path_bicycle4d + "bicycle4d_intersection_ctrl_acc_t_3.00.npy"

        elif self.scenario == "roundabout":
            self.valfunc_path_bicycle4d = self.data_path_bicycle4d + "bicycle4d_brs_roundabout_t_3.00.npy"
            self.beta_r_path_bicycle4d = self.data_path_bicycle4d + "bicycle4d_roundabout_ctrl_beta_t_3.00.npy"
            self.a_r_path_bicycle4d = self.data_path_bicycle4d + "bicycle4d_roundabout_ctrl_acc_t_3.00.npy"

        self.safe_data["bicycle4d"] = {}
        # Previous interpolation
        self.safe_data["bicycle4d"]["valfunc"] = np.load(self.valfunc_path_bicycle4d)
        self.safe_data["bicycle4d"]["beta_r"] = np.load(self.beta_r_path_bicycle4d)
        self.safe_data["bicycle4d"]["a_r"] = np.load(self.a_r_path_bicycle4d)

        # load Reldyn5D data ####################################################################################################
        self.safe_data["reldyn5d"] = {}

        self.data_path_reldyn5d = "/home/anjianl/Desktop/project/optimized_dp/data/brs/1006-correct/"

        # Choose valfunc and ctrl (beta_r, a_r)
        for mode in range(-1, 6):
            self.valfunc_path_reldyn5d = self.data_path_reldyn5d + "mode{:d}/reldyn5d_brs_mode{:d}_t_3.00.npy".format(mode,
                                                                                                    mode)
            self.beta_r_path_reldyn5d = self.data_path_reldyn5d + "mode{:d}/reldyn5d_ctrl_beta_mode{:d}_t_3.00.npy".format(mode,
                                                                                                         mode)
            self.a_r_path_reldyn5d = self.data_path_reldyn5d + "mode{:d}/reldyn5d_ctrl_acc_mode{:d}_t_3.00.npy".format(mode,
                                                                                                     mode)

            self.safe_data["reldyn5d"]["reldyn5d_brs_mode{:d}".format(mode)] = np.load(self.valfunc_path_reldyn5d)
            self.safe_data["reldyn5d"]["reldyn5d_ctrl_beta_mode{:d}".format(mode)] = np.load(self.beta_r_path_reldyn5d)
            self.safe_data["reldyn5d"]["reldyn5d_ctrl_acc_mode{:d}".format(mode)] = np.load(self.a_r_path_reldyn5d)

    def save_data_to_json(self, filename):

        avg_max_devation = sum(self.max_deviation_list) / len(self.max_deviation_list)
        avg_min_distance = sum(self.min_dist_list) / len(self.min_dist_list)
        avg_reldyn5d_control_time = sum(self.time_use_reldyn5d_control) / len(self.time_use_reldyn5d_control)
        avg_bicycle4d_control_time = sum(self.time_use_bicycl4d_control) / len(self.time_use_bicycl4d_control)

        avg_final_target_index = sum(self.final_target_index_list) / len(self.final_target_index_list)

        print("average max deviation is", avg_max_devation)
        print("average min distance is", avg_min_distance)
        print("average reldyn5d control is", avg_reldyn5d_control_time)
        print("average bicycle4d control is", avg_bicycle4d_control_time)

        collision_time_05 = sum(i < 0.5 for i in self.min_dist_list)
        collision_time_10 = sum(i < 1.0 for i in self.min_dist_list)

        not_recovery_time_05 = sum(i > 0.5 for i in self.final_deviation_list)
        not_recovery_time_10 = sum(i > 1.0 for i in self.final_deviation_list)

        # Write statistics to json file #######################################################################
        data = {}

        data['episode_para'] = []
        data['episode_para'].append({
            'episode_num': int(self.range_radius * 2),
            'robot_target_speed': self.robot_target_speed
        })

        data['deviation'] = []
        data['deviation'].append({
            'avg_max_deviation': avg_max_devation,
            'up_max_deviation': np.max(self.max_deviation_list),
            'low_max_deviation': np.min(self.max_deviation_list)
        })

        data['distance'] = []
        data['distance'].append({
            'avg_min_distance': avg_min_distance,
            'low_min_distance': np.min(self.min_dist_list)
        })

        data['safe_control'] = []
        data['safe_control'].append({
            'avg_reldyn5d_control_time': avg_reldyn5d_control_time,
            'avg_bicycle4d_control_time': avg_bicycle4d_control_time
        })

        data['collision'] = []
        data['collision'].append({
            'less_than_0.5': int(collision_time_05),
            'less_then_1.0': int(collision_time_10)
        })

        data['task_complete'] = []
        data['task_complete'].append({
            'final_target_index': avg_final_target_index
        })

        data['recovery'] = []
        data['recovery'].append({
            'final_deviation_larger_than_0.5': int(not_recovery_time_05),
            'final_deviation_larger_than_1.0': int(not_recovery_time_10)
        })

        file_dir = self.statistics_file_dir + '/' + filename + '/'
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        if self.use_safe_control:
            if self.use_prediction:
                with open(file_dir + filename + "_pred.json", 'w') as outfile:
                    # json.dump(data, outfile)
                    str_ = json.dumps(data, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
                    outfile.write(str(str_))
                    print("With prediction statistics saved!")
            else:
                with open(file_dir + filename + "_no_pred.json", 'w') as outfile:
                    # json.dump(data, outfile)
                    str_ = json.dumps(data, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
                    outfile.write(str(str_))
                    print("no prediction statistics saved!")
        else:
            with open(file_dir + filename + "_no_safe_ctrl.json", 'w') as outfile:
                # json.dump(data, outfile)
                str_ = json.dumps(data, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
                outfile.write(str(str_))
                print("no safe control statistics saved!")

    def reset_trial(self, trial_name, scenario):

        # reset parameters ##############################################################################################
        if scenario == "intersection":

            # self.show_animation = True
            self.show_animation = False

            # self.save_plot = True
            self.save_plot = False
            self.fig_save_dir = "/home/anjianl/Desktop/project/optimized_dp/result/simulation/1013/intersection/"
            self.figure_file_name = "trial_1"

            # Save statistics
            self.save_statistics = True
            # self.save_statistics = False
            self.statistics_file_dir = "/home/anjianl/Desktop/project/optimized_dp/result/statistics/1013/intersection/"

        elif scenario == "roundabout":

            # self.show_animation = True
            self.show_animation = False

            # self.save_plot = True
            self.save_plot = False
            self.fig_save_dir = "/home/anjianl/Desktop/project/optimized_dp/result/simulation/1013/roundabout/"
            self.figure_file_name = "trial_1"

            # Save statistics
            self.save_statistics = True
            # self.save_statistics = False
            self.statistics_file_dir = "/home/anjianl/Desktop/project/optimized_dp/result/statistics/1013/roundabout/"

        # reset trial trajectory ##############################################################################################
        if scenario == "intersection":
            if trial_name == "trial_1":
                self.huamn_car_file_name_intersection = 'car_36_vid_11.csv'
                self.robot_car_file_name_intersection = 'car_20_vid_09_refPath.csv'
                self.human_start_step = 164
                self.robot_target_speed = 2
                self.curr_robot_start_step = 67
                self.max_t = 10
                self.statistics_file_name = "h_36_r_20_stanley"
                self.range_radius = 5
            elif trial_name == "trial_2":
                self.huamn_car_file_name_intersection = 'car_36_vid_11.csv'
                self.robot_car_file_name_intersection = 'car_52_vid_07_refPath.csv'
                self.human_start_step = 170
                self.robot_target_speed = 2
                # Use full set is even worse than don't use safe control
                # self.robot_start_step = 42
                # Use safe control is both good
                self.curr_robot_start_step = 48
                self.range_radius = 10
                self.statistics_file_name = "h_36_r_52_stanley"
                self.max_t = 10
            elif trial_name == "trial_3": # TODO: trail 3 bad cases, with prediction it has large deviation
                self.huamn_car_file_name_intersection = 'car_20_vid_09.csv'
                self.robot_car_file_name_intersection = 'car_36_vid_11_refPath.csv'
                self.human_start_step = 200
                self.robot_target_speed = 2
                self.curr_robot_start_step = 70
                self.max_t = 10
                self.statistics_file_name = "h_20_r_36_stanley"
                self.range_radius = 10
        elif scenario == "roundabout":
            if trial_name == "trial_1":
                self.huamn_car_file_name_roundabout = 'car_41.csv'
                self.robot_car_file_name_roundabout = 'car_36_refPath.csv'
                self.human_start_step = 30
                self.robot_target_speed = 2
                self.curr_robot_start_step = 58
                self.max_t = 10
                self.statistics_file_name = "h_41_r_36_stanley"
                self.range_radius = 10
            elif trial_name == "trial_2":
                self.huamn_car_file_name_roundabout = 'car_29.csv'
                self.robot_car_file_name_roundabout = 'car_30_refPath.csv'
                self.human_start_step = 0
                self.robot_target_speed = 2
                self.curr_robot_start_step = 57
                self.max_t = 10
                self.statistics_file_name = "h_29_r_30_stanley"
                self.range_radius = 10
            elif trial_name == "trial_3": # TODO: trial 3, bad example
                self.huamn_car_file_name_roundabout = 'car_24.csv'
                self.robot_car_file_name_roundabout = 'car_6_refPath.csv'
                self.human_start_step = 31
                self.robot_target_speed = 2
                self.curr_robot_start_step = 45
                self.max_t = 10
                self.statistics_file_name = "h_24_r_6_stanley"
                self.range_radius = 10


    def main(self):

        start_time = time.time()

        ###################################################################################################
        ########################### Simulation over all options #################################################
        # Loop over all options
        for scenario in ["intersection", "roundabout"]:
            # Configure scenario and load data
            if scenario == "intersection":
                self.scenario = "intersection"
                self.load_safe_data()
            elif scenario == "roundabout":
                self.scenario = 'roundabout'
                self.load_safe_data()

            # Loop over different trial and conditions
            for trial in ["trial_1", "trial_2", "trial_3"]:
                # Reset trial trajectory file
                self.reset_trial(trial_name=trial, scenario=scenario)

                for use_safe_control in [True, False]:
                    for use_prediction in [True, False]:

                        # Have a list to store statistics
                        self.min_dist_list = []
                        self.max_deviation_list = []
                        self.time_use_reldyn5d_control = []
                        self.time_use_bicycl4d_control = []
                        self.final_target_index_list = []
                        self.final_deviation_list = []

                        # Assign parameters
                        self.use_safe_control = use_safe_control
                        self.use_prediction = use_prediction

                        if (not self.use_safe_control) and (self.use_prediction):
                            continue

                        for i in range(self.curr_robot_start_step - self.range_radius, self.curr_robot_start_step + self.range_radius + 1):
                            self.robot_start_step = i
                            self.simulate()

                        # Save data to json file
                        if self.save_statistics:
                            self.save_data_to_json(filename=self.statistics_file_name)

        end_time = time.time()
        print("whole simulation takes ", end_time - start_time)

        return 0


if __name__ == "__main__":
    SimulatorStanleyV2().main()