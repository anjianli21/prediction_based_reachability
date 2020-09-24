import numpy as np
import sys
sys.path.append("/Users/anjianli/Desktop/robotics/project/optimized_dp")
from matplotlib import pyplot as plt
from matplotlib import animation, rc

import pandas
import scipy.io
from scipy import integrate

from simulation.optimal_control import OptimalControl
from prediction.process_prediction_v3 import ProcessPredictionV3

class Simulator(object):

    """
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

        self.scenario = "intersection"

        if '/Users/anjianli/anaconda3/envs/hcl-env/lib/python3.8' not in sys.path:
            self.file_dir_intersection = '/home/anjianl/Desktop/project/optimized_dp/data/intersection-data'
            self.file_dir_roundabout = '/home/anjianl/Desktop/project/optimized_dp/data/roundabout-data'
        else:
            # My laptop
            self.file_dir_intersection = '/Users/anjianli/Desktop/robotics/project/optimized_dp/data/intersection-data'
            self.file_dir_roundabout = '/Users/anjianli/Desktop/robotics/project/optimized_dp/data/roundabout-data'

        self.huamn_car_file_name_intersection = 'car_20_vid_09.csv'
        self.robot_car_file_name_intersection = 'car_36_vid_11_refPath.csv'


    def simulate(self):

        print("Start simulating")

        # Set time
        t0 = 0
        tMax = 100
        dt = 0.1
        # Or step num?
        step_num = 200

        # Read prediction as human car's trajectory
        human_car_traj = self.get_traj_from_prediction(filename=self.huamn_car_file_name_intersection)

        # Read ref path as robot_car_traj
        robot_car_traj = self.get_traj_from_ref_path(filename=self.robot_car_file_name_intersection)

        # Plot current traj
        self.plot(robot_car_traj, human_car_traj)

        # Init human traj for simulation
        human_init_step = 40
        human_init = [[human_car_traj["x_t"][human_init_step], human_car_traj["y_t"][human_init_step]]]
        simulation_human_traj = human_init

        # Init robot traj for simulation
        robot_init_step = 0
        simulation_robot_traj = [[robot_car_traj["x_t"][robot_init_step], robot_car_traj["y_t"][robot_init_step]]]

        for i in range(step_num):
            time = t0 + step_num * dt

            # Retrieve current states
            curr_human_states = simulation_human_traj[-1]
            curr_robot_states = simulation_robot_traj[-1]

            # Estimate the state of robot and human
            x_h, y_h, psi_h, v_h = self.get_states_from_xy()
            x_r, y_r, psi_r, v_r = self.get_states_from_xy()

            # Get predicted trajectory of human car
            x_h_pred, y_h_pred = self.get_human_car_prediction()

            # Get reachability value function and optimal control for robot car
            val_func, optctrl_beta_r, optctrl_a_r = OptimalControl()

            # Update robot states
            if val_func > 0:
                # Check if it's on ref path:
                if self.on_ref_path():
                    # Only plan speed
                    plan_speed = 1
                else:
                    # if not on ref path, return to ref path.
                    return_to_ref = 1
            else:
                # If inside, reachable set, use optimal control to integrate the states
                robot_next_step = integrate.odeint()

            # Update human states, just append the next prediction
            human_next_step = 0

            # Append the next states on simulation trajectory
            simulation_human_traj.append(human_next_step)
            simulation_robot_traj.append(robot_next_step)


    def get_traj_from_prediction(self, filename):

        # Read trajectory from prediction
        if self.scenario == "intersection":
            traj_file_name = self.file_dir_intersection + '/' + filename
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
        if self.scenario == "intersection":
            traj_file_name = self.file_dir_intersection + '/' + filename
        else:
            traj_file_name = ""

        traj_file = pandas.read_csv(traj_file_name, header=None, usecols=[0, 1], names=['x_t', 'y_t'],)
        length = len(traj_file)

        robot_car_traj = {}
        robot_car_traj['x_t'] = np.asarray(traj_file["x_t"]).tolist()
        robot_car_traj['y_t'] = np.asarray(traj_file["y_t"]).tolist()

        return robot_car_traj

    def plot(self, robot_car_traj, human_car_traj):
        robot_car_x = robot_car_traj["x_t"]
        robot_car_y = robot_car_traj["y_t"]

        human_car_x = human_car_traj["x_t"]
        human_car_y = human_car_traj["y_t"]

        # intersection_curbs = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/intersection_curbs.npy")
        intersection_curbs = np.load(
            "/Users/anjianli/Desktop/robotics/project/optimized_dp/data/map/obstacle_map/intersection_curbs.npy")

        fig, ax = plt.subplots()

        ax.axis('equal')

        ax.scatter(intersection_curbs[0], intersection_curbs[1], color='black', linewidths=0.3)

        ims = []
        for i in range(len(human_car_x)):
            # ax.set_title('t = {:.2f}'.format(i * 0.1))
            im = ax.scatter(human_car_x[45:i + 45], human_car_y[45:i + 45], animated=True, marker="o", color='red')
            im2 = ax.scatter(robot_car_x[:i + 1], robot_car_y[:i + 1], animated=True, color='green')
            ims.append([im, im2])

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                        repeat_delay=1000, repeat=False)

        plt.show()

if __name__ == "__main__":

    Simulator().simulate()
