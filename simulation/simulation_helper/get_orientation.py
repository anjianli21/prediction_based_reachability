import numpy as np
import pandas
import copy
import sys
sys.path.append("/Users/anjianli/Desktop/robotics/project/optimized_dp")

import matplotlib.pyplot as plt

from prediction.process_prediction_v3 import ProcessPredictionV3

class GetOrientation(object):

    """
    Here we pre-compute the orientation of the human car, and save then into .csv file.
    Then in the simulation, we can directly get the smooth orientation of the human car in every time step

    """

    def __init__(self):

        # Data directory
        # Remote desktop
        if '/Users/anjianli/anaconda3/envs/hcl-env/lib/python3.8' not in sys.path:
            self.file_dir_intersection = '/home/anjianl/Desktop/project/optimized_dp/data/intersection-data'
            self.file_dir_roundabout = '/home/anjianl/Desktop/project/optimized_dp/data/roundabout-data'
            self.file_dir_intersection_new = '/home/anjianl/Desktop/project/optimized_dp/data/intersection-data-psi'
            self.file_dir_roundabout_new = '/home/anjianl/Desktop/project/optimized_dp/data/roundabout-data-psi'
        else:
            # My laptop
            self.file_dir_intersection = '/Users/anjianli/Desktop/robotics/project/optimized_dp/data/intersection-data'
            self.file_dir_roundabout = '/Users/anjianli/Desktop/robotics/project/optimized_dp/data/roundabout-data'
            self.file_dir_intersection_new = '/Users/anjianli/Desktop/robotics/project/optimized_dp/data/intersection-data-psi'

        # File name
        self.file_name_intersection = ['car_16_vid_09.csv', 'car_20_vid_09.csv', 'car_29_vid_09.csv',
                                       'car_36_vid_11.csv', 'car_50_vid_03.csv', 'car_112_vid_11.csv',
                                       'car_122_vid_11.csv',
                                       'car_38_vid_02.csv', 'car_52_vid_07.csv', 'car_73_vid_02.csv',
                                       'car_118_vid_11.csv']
        self.file_name_roundabout = ['car_27.csv', 'car_122.csv',
                                     'car_51.csv', 'car_52.csv', 'car_131.csv', 'car_155.csv',
                                     'car_15.csv', 'car_28.csv', 'car_34.csv', 'car_41.csv', 'car_50.csv',
                                     'car_61.csv', 'car_75.csv', 'car_80.csv',
                                     'car_2.csv', 'car_3.csv', 'car_4.csv', 'car_6.csv', 'car_9.csv', 'car_10.csv',
                                     'car_11.csv', 'car_13.csv', 'car_17.csv', 'car_21.csv', 'car_24.csv', 'car_29.csv',
                                     'car_30.csv', 'car_35.csv', 'car_36.csv', 'car_44.csv']

        # Fit polynomial
        self.degree = 5

        self.time_step = 0.1


    def get_orientation(self):

        # filename_list = copy.copy(self.file_name_intersection)
        # file_dir = copy.copy(self.file_dir_intersection)

        filename_list = copy.copy(self.file_name_roundabout)
        file_dir = copy.copy(self.file_dir_roundabout)

        for file_name in filename_list:
            full_file_name = file_dir + '/' + file_name

            traj_file = self.read_prediction(file_name=full_file_name)

            # Extract trajectory segments from csv files
            raw_traj = self.extract_traj(traj_file)

            # Fit polynomial for x, y position: x(t), y(t)
            poly_traj = self.fit_polynomial_traj(raw_traj)

            # Get orientation for each segment of predictions
            orientation_list = self.compute_orientation(poly_traj)

            # Write the orientation in new files
            new_file_name = self.file_dir_roundabout_new + '/' + file_name
            self.save_orientation(orientation_list, new_file_name)

        return 0

    def save_orientation(self, orientation_list, new_file_name):

        whole_orientation_list = []

        for orientation in orientation_list:
            whole_orientation_list.extend(orientation.tolist())

        df = pandas.read_csv(new_file_name)

        df['psi_t'] = whole_orientation_list

        df.to_csv(new_file_name, index=False)

    def compute_orientation(self, poly_traj):

        psi_list = []

        for poly_traj_seg in poly_traj:

            length = len(poly_traj_seg['x_t_poly'])
            # Directly compute dx/dt, dy/dt
            dx_t = (poly_traj_seg['x_t_poly'][1:length] - poly_traj_seg['x_t_poly'][0:(length - 1)]) / self.time_step
            dy_t = (poly_traj_seg['y_t_poly'][1:length] - poly_traj_seg['y_t_poly'][0:(length - 1)]) / self.time_step

            psi = np.arctan2(dy_t, dx_t)
            psi = np.append(psi, psi[-1])
            psi_list.append(psi)

        # If we fit polynomial for the whole trajectory, instead of each prediction

        return psi_list

    def read_prediction(self, file_name=None):
        """
        Read CSV file that contains predicted goal states and generated trajectory

        """

        # prediction_file = pandas.read_csv(self.file_dir + '/' + self.file_name)
        prediction_file = pandas.read_csv(file_name)

        return prediction_file

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
            degree = self.degree
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

if __name__ == "__main__":

    GetOrientation().get_orientation()