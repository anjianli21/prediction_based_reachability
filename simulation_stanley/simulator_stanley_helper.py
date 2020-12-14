import numpy as np

import pandas

from prediction.predict_mode import PredictMode
from prediction.process_prediction import ProcessPrediction

class SimulatorStanleyHelper(object):

    def __init__(self):

        self.file_dir_intersection_psi = "/home/anjianl/Desktop/project/optimized_dp/data/intersection-data-psi"
        self.file_dir_roundabout_psi = "/home/anjianl/Desktop/project/optimized_dp/data/roundabout-data-psi"

        self.file_dir_intersection = "/home/anjianl/Desktop/project/optimized_dp/data/intersection-data"
        self.file_dir_roundabout = "/home/anjianl/Desktop/project/optimized_dp/data/roundabout-data"

        # Configure curbs map location
        self.intersection_curbs = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/intersection_curbs.npy")
        self.roundabout_curbs = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/roundabout_curbs.npy")

    def get_traj_from_prediction(self, scenario, filename):

        # Read trajectory from prediction
        if scenario == "intersection":
            traj_file_name = self.file_dir_intersection_psi + '/' + filename
        elif scenario == "roundabout":
            traj_file_name = self.file_dir_roundabout_psi + '/' + filename
        traj_file = pandas.read_csv(traj_file_name)
        length = len(traj_file)

        raw_traj = []
        traj_seg = {}
        traj_seg['x_t'] = []
        traj_seg['y_t'] = []
        traj_seg['v_t'] = []
        traj_seg['psi_t'] = []

        for i in range(length):
            traj_seg['x_t'].append(traj_file['x_t'][i])
            traj_seg['y_t'].append(traj_file['y_t'][i])
            traj_seg['v_t'].append(traj_file['v_t'][i])
            traj_seg['psi_t'].append(traj_file['psi_t'][i])
            if traj_file['t_to_goal'][i] == 0:
                raw_traj.append(traj_seg)
                traj_seg = {}
                traj_seg['x_t'] = []
                traj_seg['y_t'] = []
                traj_seg['v_t'] = []
                traj_seg['psi_t'] = []
        # print(raw_traj)

        # raw_trajectory is cut into several piece. We can concatenate them together
        human_car_traj = {"x_t": [], "y_t": [], "v_t": [], "psi_t": []}
        for i in range(len(raw_traj)):
            human_car_traj["x_t"] += raw_traj[i]["x_t"]
            human_car_traj["y_t"] += raw_traj[i]["y_t"]
            human_car_traj["v_t"] += raw_traj[i]["v_t"]
            human_car_traj["psi_t"] += raw_traj[i]["psi_t"]

        if scenario == "roundabout":
            human_car_traj['x_t'] = [x + 1000 for x in human_car_traj['x_t']]  # TODO: modify human car traj x
            human_car_traj['y_t'] = [y + 1000 for y in human_car_traj['y_t']]  # TODO: modify human car traj y

        return human_car_traj

    def get_traj_from_ref_path(self, scenario, filename):

        # Read trajectory from prediction
        if scenario == "intersection":
            traj_file_name = self.file_dir_intersection + '/' + filename
        elif scenario == "roundabout":
            traj_file_name = self.file_dir_roundabout + '/' + filename

        traj_file = pandas.read_csv(traj_file_name, header=None, usecols=[0, 1], names=['x_t', 'y_t'],)
        length = len(traj_file)

        robot_car_traj = {}
        robot_car_traj['x_t'] = np.asarray(traj_file["x_t"]).tolist()
        robot_car_traj['y_t'] = np.asarray(traj_file["y_t"]).tolist()

        if scenario == "roundabout":
            robot_car_traj['x_t'] = [x + 1000 for x in robot_car_traj['x_t']]  # TODO: modify robot car traj x
            robot_car_traj['y_t'] = [y + 1000 for y in robot_car_traj['y_t']]  # TODO: modify robot car traj y

        return robot_car_traj

    def init_robot_state(self, robot_car_traj_ref, robot_start_step):

        dx = (robot_car_traj_ref['x_t'][robot_start_step+1] - robot_car_traj_ref['x_t'][robot_start_step]) / 0.1
        dy = (robot_car_traj_ref['y_t'][robot_start_step+1] - robot_car_traj_ref['y_t'][robot_start_step]) / 0.1
        x_r_init, y_r_init = robot_car_traj_ref['x_t'][robot_start_step], robot_car_traj_ref['y_t'][robot_start_step]
        v_r_init = np.sqrt(dx ** 2 + dy ** 2)
        psi_r_init = np.arctan2(dy, dx)

        return x_r_init, y_r_init, psi_r_init, v_r_init

    def get_human_car_prediction(self, human_car_traj, curr_step_human, episode_len):

        traj_to_pred = {}
        if curr_step_human + episode_len < len(human_car_traj['x_t']) - 1:
            traj_to_pred['psi_t'] = np.asarray(human_car_traj['psi_t'][curr_step_human:curr_step_human + episode_len])
            traj_to_pred['v_t'] = np.asarray(human_car_traj['v_t'][curr_step_human:curr_step_human + episode_len])
        else:
            traj_to_pred['psi_t'] = np.asarray(human_car_traj['psi_t'][curr_step_human:])
            traj_to_pred['v_t'] = np.asarray(human_car_traj['v_t'][curr_step_human:])

        length = len(traj_to_pred['v_t']) - 1
        # raw_acc_list, raw_omega_list = ProcessPrediction().get_action_v_profile([traj_to_pred], poly_traj)
        raw_acc_list = [(traj_to_pred['v_t'][1:length] - traj_to_pred['v_t'][0:length - 1]) / ProcessPrediction().time_step]
        raw_omega_list = [(traj_to_pred['psi_t'][1:length] - traj_to_pred['psi_t'][0:length - 1]) / ProcessPrediction().time_step]

        filter_acc_list, filter_omega_list = PredictMode().filter_action(raw_acc_list, raw_omega_list)

        filter_acc, filter_omega = filter_acc_list[0], filter_omega_list[0]

        mode_num_list, mode_probability_list = PredictMode().get_mode(filter_acc, filter_omega)

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