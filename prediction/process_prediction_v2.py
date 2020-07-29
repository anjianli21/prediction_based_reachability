import numpy as np
import pandas
import matplotlib.pyplot as plt
import csv


class ProcessPredictionV2(object):

    def __init__(self):

        self.file_dir_intersection = '/home/anjianl/Desktop/project/optimized_dp/data/csv_files_for_planner/intersection'
        self.file_name_intersection = ['car_16_vid_09.csv', 'car_20_vid_09.csv', 'car_29_vid_09.csv',
                                       'car_36_vid_11.csv', 'car_50_vid_03.csv', 'car_112_vid_11.csv',
                                       'car_122_vid_11.csv']
        self.file_dir_roundabout = '/home/anjianl/Desktop/project/optimized_dp/data/csv_files_for_planner/roundabout'
        self.file_name_roundabout = ['car_27.csv', 'car_122.csv']

        self.time_step = 0.1
        self.time_filter = 10

    def read_prediction(self, file_name=None):
        """
        Read CSV file that contains predicted goal states and generated trajectory

        """

        # prediction_file = pandas.read_csv(self.file_dir + '/' + self.file_name)
        prediction_file = pandas.read_csv(file_name)

        # print(prediction_file)
        #
        # print(prediction_file['v_t'])


        return prediction_file

    def get_action_data(self, file_name=None):
        """
        Extract acceleration and orientation speed for predicted trajectory

        """

        traj_file = self.read_prediction(file_name=file_name)

        # Extract trajectory segments from csv files
        raw_traj = self.extract_traj(traj_file)

        # Fit polynomial for x, y position: x(t), y(t)
        poly_traj = self.fit_polynomial_traj(raw_traj)

        # Get actions from poly_traj
        acc_list, omega_list = self.get_action(poly_traj)

        return acc_list, omega_list

    def extract_traj(self, traj_file):
        """
        Extract trajectory segments from csv file

        :param traj_file:
        :return:
        """
        length = len(traj_file)

        raw_traj = []
        traj_seg = {}
        traj_seg['x_t'] = []
        traj_seg['y_t'] = []

        for i in range(length):
            traj_seg['x_t'].append(traj_file['x_t'][i])
            traj_seg['y_t'].append(traj_file['y_t'][i])
            if traj_file['t_to_goal'][i] == 0:
                raw_traj.append(traj_seg)
                traj_seg = {}
                traj_seg['x_t'] = []
                traj_seg['y_t'] = []

        return raw_traj

    def fit_polynomial_traj(self, raw_traj):
        """
        Given a trajectory segment, fit a polynomial
        :param raw_traj:
        :return:
        """

        poly_traj = []

        for raw_traj_seg in raw_traj:
            t = np.asarray(range(len(raw_traj_seg['x_t'])))
            x_t = np.asarray(raw_traj_seg['x_t'])
            y_t = np.asarray(raw_traj_seg['y_t'])

            # Fit a 5-degree polynomial
            degree = 5
            weights_x = np.polyfit(t, x_t, degree)
            weights_y = np.polyfit(t, y_t, degree)

            poly_x_func = np.poly1d(weights_x)
            poly_x_t = np.asarray([poly_x_func(t) for t in range(len(t))])
            poly_y_func = np.poly1d(weights_y)
            poly_y_t = np.asarray([poly_y_func(t) for t in range(len(t))])

            # Plot the comparison
            # plt.plot(t, x_t)
            # plt.plot(t, poly_x_t)
            # plt.show()

            # Form the polynomial trajectory list
            poly_traj_seg = {}
            poly_traj_seg['x_t_poly'] = poly_x_t
            poly_traj_seg['y_t_poly'] = poly_y_t
            poly_traj.append(poly_traj_seg)

        return poly_traj

    def get_action(self, poly_traj):
        """
        Extract acceleration and omega (angular speed) for each generated trajectory

        """

        acceleration_list = []
        omega_list = []

        for poly_traj_seg in poly_traj:

            length = len(poly_traj_seg['x_t_poly'])
            dx_t = (poly_traj_seg['x_t_poly'][1:length] - poly_traj_seg['x_t_poly'][0:(length - 1)]) / self.time_step
            dy_t = (poly_traj_seg['y_t_poly'][1:length] - poly_traj_seg['y_t_poly'][0:(length - 1)]) / self.time_step

            dx1_t = (poly_traj_seg['x_t_poly'][1:(length - 1)] - poly_traj_seg['x_t_poly'][0:(length - 2)]) / self.time_step
            dx2_t = (poly_traj_seg['x_t_poly'][2:length] - poly_traj_seg['x_t_poly'][1:(length - 1)]) / self.time_step
            dy1_t = (poly_traj_seg['y_t_poly'][1:(length - 1)] - poly_traj_seg['y_t_poly'][0:(length - 2)]) / self.time_step
            dy2_t = (poly_traj_seg['y_t_poly'][2:length] - poly_traj_seg['y_t_poly'][1:(length - 1)]) / self.time_step

            # Get acceleration
            v_t = np.sqrt(dx_t ** 2 + dy_t ** 2)
            a_t = (v_t[1:len(v_t)] - v_t[0:(len(v_t) - 1)]) / self.time_step
            acceleration_list.append(a_t)

            # Get omega (angular speed)
            orientation_1 = np.arctan2(dy1_t, dx1_t)
            orientation_2 = np.arctan2(dy2_t, dx2_t)
            omega_t = (orientation_2 - orientation_1) / self.time_step
            omega_list.append(omega_t)

            if len(a_t) != len(omega_t):
                print("a_t and omega_t have different dimensions")
                raise SystemExit(0)

        return acceleration_list, omega_list

    def collect_action_from_group(self):

        # in the format of: (file, a_upper, a_lower, ang_v_upper, ang_v_lower)
        action_list_intersection = []
        action_list_roundabout = []

        # For intersection scenario
        for file_name in self.file_name_intersection:
            full_file_name = self.file_dir_intersection + '/' + file_name
            acc, omega = self.get_action_data(file_name=full_file_name)
            for index in range(len(acc)):
                action_list_intersection.append([file_name, acc[index], omega[index]])

        # For roundabout scenario
        for file_name in self.file_name_roundabout:
            full_file_name = self.file_dir_roundabout + '/' + file_name
            acc, omega = self.get_action_data(file_name=full_file_name)
            for index in range(len(acc)):
                action_list_roundabout.append([file_name, acc[index], omega[index]])

        return action_list_intersection, action_list_roundabout

    def get_ang_diff(self, target, base):
        """
        Given target and base both in [-pi, pi], calculate signed angle difference of (target - base)

        """

        # If either target or base == 0, then return 0
        if target == 0 or base == 0:
            return 0
        elif target > base:
            if target - base <= np.pi:
                return target - base
            else:
                return target - (base + 2 * np.pi)
        elif target == base:
            return 0
        else:
            return - self.get_ang_diff(base, target)

        # # Doesn't consider target or base == 0
        # if target > base:
        #     if target - base <= np.pi:
        #         return target - base
        #     else:
        #         return target - (base + 2 * np.pi)
        # elif target == base:
        #     return 0
        # else:
        #     return - self.get_ang_diff(base, target)

if __name__ == "__main__":
    ProcessPredictionV2().collect_action_from_group()
