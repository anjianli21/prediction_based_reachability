import sys
sys.path.append("/home/anjianl/Desktop/project/optimized_dp/")
import pandas
import os
import shutil
from PIL import Image

from simulation_lqr.lqr_speed_steer_control import *
from simulation_lqr.simulator_lqr_helper import SimulatorLQRHelper
from simulation_lqr.human_car import HumanState
from simulation_lqr.optimal_control_reldyn5d import OptimalControlRelDyn5D
from simulation_lqr.optimal_control_bicycle4d import OptimalControlBicycle4D

class SimulatorLQR(SimulatorLQRHelper):

    def __init__(self):
        super(SimulatorLQR, self).__init__()

        # Update frequency for the prediction, dt = 0.1s
        self.mode_predict_span = 1
        # When making prediction, choose trajectory over some episode
        self.episode_len = 12

        self.intersection_curbs = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/intersection_curbs.npy")
        self.roundabout_curbs = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/roundabout_curbs.npy")

    def simulate(self):

        print("Start simulating")

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
        speed_profile = calc_speed_profile(cyaw, self.robot_target_speed)

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
        min_deviation = 0

        # Main loop
        while curr_t <= max_t:

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
                h_drv_mode=mode_num, h_drv_mode_pro=mode_probability, use_prediction=self.use_prediction).get_optctrl()

            _, val_func_bicycle4d, optctrl_beta_r_bicycle4d, optctrl_a_r_bicycle4d = OptimalControlBicycle4D(
                robot_curr_states={'x_r': RobotCar.x, 'y_r': RobotCar.y, 'psi_r': RobotCar.yaw, 'v_r': RobotCar.v},
                scenario=self.scenario).get_optctrl()

            reachable_set_coordinate = np.asarray([
                contour_rel_coordinate[0, :] * np.cos(RobotCar.yaw) - contour_rel_coordinate[1, :] * np.sin(
                    RobotCar.yaw) + RobotCar.x,
                contour_rel_coordinate[0, :] * np.sin(RobotCar.yaw) + contour_rel_coordinate[1, :] * np.cos(
                    RobotCar.yaw) + RobotCar.y])

            # Get LQR controller for robot car ##########################################################################################
            dl, target_ind, e, e_th, ai = lqr_speed_steering_control(
                RobotCar, cx, cy, cyaw, ck, e, e_th, speed_profile, lqr_Q, lqr_R)

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
                    safe_update(RobotCar, optctrl_a_r_reldyn5d, optctrl_beta_r_reldyn5d)
                else:
                    safe_update(RobotCar, optctrl_a_r_bicycle4d, optctrl_beta_r_bicycle4d)
                    print("Use bicycle4d controller")
            else:
                RobotCar = update(RobotCar, ai, dl) # Use LQR controller

            if abs(RobotCar.v) <= robot_stop_speed:
                target_ind += 1

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
            min_deviation = max(min_deviation, abs(e))
            print("minimum deviation is", min_deviation)
            curr_min_dist = np.sqrt(rel_states['x_rel'] ** 2 + rel_states['y_rel'] ** 2)
            min_dist = min(curr_min_dist, min_dist)
            print("minimum car distance is", min_dist)

            # check goal ###########################################################################################################
            dx = RobotCar.x - robot_goal[0]
            dy = RobotCar.y - robot_goal[1]
            if math.hypot(dx, dy) <= goal_dis:
                print("Goal")
                break

            # Plot
            if target_ind % 1 == 0 and self.show_animation:
                plt.cla()
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(cx, cy, ".c", label="course")
                plt.plot(x_r_list, y_r_list, "-b", label="robot trajectory")
                plt.plot(x_h_list, y_h_list, "-r", label="human trajectory")
                plt.plot(x_r_list[-1], y_r_list[-1], "xg", label="robot pos")
                plt.plot(x_h_list[-1], y_h_list[-1], "xr", label="human pos")
                plt.plot(reachable_set_coordinate[0, :], reachable_set_coordinate[1, :], "y")
                # plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
                if self.scenario == "intersection":
                    plt.scatter(self.intersection_curbs[0], self.intersection_curbs[1], color='black', linewidths=0.03)
                elif self.scenario == "roundabout":
                    plt.scatter(self.roundabout_curbs[0], self.roundabout_curbs[1], color='black', linewidths=0.03)
                plt.axis("equal")
                plt.grid(True)
                plt.title("robot car speed (m/s):" + str(round(RobotCar.v, 2)) + ", min deviation:" + str(round(min_deviation, 2)) + ", min distance:" + str(round(min_dist, 2)))
                plt.pause(0.0001)

            if self.save_plot:
                tmp_folder_path = self.fig_save_dir + "tmp/"
                if not os.path.exists(tmp_folder_path):
                    os.mkdir(tmp_folder_path)
                plt.savefig(tmp_folder_path + "t_{:.2f}.png".format(curr_t))

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
                    file_name = self.file_name + "_with_prediction_safe_control.gif"
                else:
                    file_name = self.file_name + "_no_prediction_safe_control.gif"
            else:
                file_name = self.file_name + "_no_safe_control.gif"
            frames[0].save(
                save_dir + file_name,
                format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=300, loop=0)

            # Delete tmp dir
            shutil.rmtree(tmp_folder_path)

        # Plot after simulation finished ###########################################################################################################
        # if self.show_animation:  # pragma: no cover
        #     plt.close()
        #     plt.subplots(1)
        #     plt.plot(ax, ay, "xb", label="waypoints")
        #     plt.plot(cx, cy, "-r", label="target course")
        #     plt.plot(x_r_list, y_r_list, "-b", label="robot trajectory")
        #     plt.plot(x_h_list, y_h_list, "-r", label="human trajectory")
        #     plt.plot(x_r_list[-1], y_r_list[-1], "xg", label="robot pos")
        #     plt.plot(x_h_list[-1], y_h_list[-1], "xr", label="human pos")
        #     plt.grid(True)
        #     plt.axis("equal")
        #     plt.xlabel("x[m]")
        #     plt.ylabel("y[m]")
        #     plt.legend()
        #
        #     plt.subplots(1)
        #     plt.plot(s, [np.rad2deg(iyaw) for iyaw in cyaw], "-r", label="yaw")
        #     plt.grid(True)
        #     plt.legend()
        #     plt.xlabel("line length[m]")
        #     plt.ylabel("yaw angle[deg]")
        #
        #     plt.subplots(1)
        #     plt.plot(s, ck, "-r", label="curvature")
        #     plt.grid(True)
        #     plt.legend()
        #     plt.xlabel("line length[m]")
        #     plt.ylabel("curvature [1/m]")
        #
        #     plt.show()
        return 0

    def main(self):

        ###################################################################################################
        ########################### Intersection Scenario #################################################
        # self.scenario = "intersection"
        #
        # # Choose a trajectory reference
        # # # TODO: trial 1
        # # self.huamn_car_file_name_intersection = 'car_36_vid_11.csv'
        # # self.robot_car_file_name_intersection = 'car_20_vid_09_refPath.csv'
        # # self.human_start_step = 164
        # # self.robot_target_speed = 2
        # # self.robot_start_step = 67
        # # self.max_t = 11
        #
        # # # Trial 2 TODO: Good show of our predicion works!
        # self.huamn_car_file_name_intersection = 'car_36_vid_11.csv'
        # self.robot_car_file_name_intersection = 'car_52_vid_07_refPath.csv'
        # self.human_start_step = 170
        # self.robot_target_speed = 2
        # # Use full set is even worse than don't use safe control
        # self.robot_start_step = 42
        # # Use safe control is both good
        # # self.robot_start_step = 48
        #
        # self.max_t = 8
        #
        # # Configure parameters #############################################################################
        # self.show_animation = True
        # # self.show_animation = False
        #
        # self.use_safe_control = True
        # # self.use_safe_control = False
        #
        # self.use_prediction = True
        # # self.use_prediction = False
        #
        # # self.save_plot = True
        # self.save_plot = False
        #
        # self.fig_save_dir = "/home/anjianl/Desktop/project/optimized_dp/result/simulation/1009/intersection/"
        # self.file_name = "trial_2"
        # self.simulate()

        ###################################################################################################
        ########################### Roundabout Scenario #################################################
        self.scenario = "roundabout"

        # Choose a trajectory reference
        # # TODO: trial 1
        # self.huamn_car_file_name_roundabout = 'car_41.csv'
        # self.robot_car_file_name_roundabout = 'car_36_refPath.csv'
        # self.human_start_step = 30
        # self.robot_target_speed = 2
        # self.robot_start_step = 52
        # self.robot_start_step = 53
        # self.max_t = 7
        # self.file_name = "trial_1"

        ## TODO: trial 2
        self.huamn_car_file_name_roundabout = 'car_29.csv'
        self.robot_car_file_name_roundabout = 'car_30_refPath.csv'
        self.human_start_step = 0
        self.robot_target_speed = 2
        self.robot_start_step = 57
        # self.robot_start_step = 58

        self.max_t = 9
        self.file_name = "trial_2"

        # Configure parameters #############################################################################
        self.show_animation = True
        # self.show_animation = False

        self.use_safe_control = True
        # self.use_safe_control = False

        self.use_prediction = True
        # self.use_prediction = False

        self.save_plot = True
        # self.save_plot = False

        self.fig_save_dir = "/home/anjianl/Desktop/project/optimized_dp/result/simulation/1009/roundabout/"

        self.simulate()

        return 0


if __name__ == "__main__":
    SimulatorLQR().main()