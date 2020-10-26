import sys
sys.path.append("/home/anjianl/Desktop/project/optimized_dp")

from plot_utils import map_vis_without_lanelet

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle

from matplotlib.widgets import Button

from plot_utils import dataset_reader
from plot_utils import dataset_types
from plot_utils import map_vis_lanelet2
from plot_utils import tracks_vis
from plot_utils import dict_utils

class PlotTrajectory(object):

    def __init__(self):

        # self.use_prediction = True
        # self.use_prediction = False

        # self.scenario = "roundabout"
        self.scenario = "intersection"

        self.car_length = 2.8
        self.car_width = 1

    def main(self):

        self.load_data()

        fig, axes = plt.subplots(1, 1)

        # load and draw the lanelet2 map, either with or without the lanelet2 library
        lat_origin = 0.  # origin is necessary to correctly project the lat lon values in the osm file to the local
        lon_origin = 0.  # coordinates in which the tracks are provided; we decided to use (0|0) for every scenario
        print("Loading map...")

        if self.scenario == "intersection":
            lanelet_map_file = "/home/anjianl/Desktop/project/optimized_dp/data/map/osm/DR_USA_Intersection_EP0.osm"
        elif self.scenario == "roundabout":
            lanelet_map_file = "/home/anjianl/Desktop/project/optimized_dp/data/map/osm/DR_USA_Roundabout_FT.osm"

        ######################################### Main loop #########################################################################

        for i in range(len(self.human_traj["t"])):

            plt.cla()
            map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, lat_origin, lon_origin)
            axes.plot(self.human_traj["x_h"][:i], self.human_traj["y_h"][:i], "-r", label="human trajectory", linewidth=0.5)
            axes.plot(self.robot_traj_pred["x_r"][:i], self.robot_traj_pred["y_r"][:i], "-b", label="robot trajectory prediction", linewidth=0.5)  # with prediction
            axes.plot(self.robot_traj_nopred["x_r"][:i], self.robot_traj_nopred["y_r"][:i], "-g", label="robot trajectory no prediction", linewidth=0.5)  # no prediction
            # plt.plot(cx, cy, ".c", label="course")

            robot_pred_rect = matplotlib.patches.Polygon(self.polygon_xy_from_motionstate(x=self.robot_traj_pred["x_r"][i],
                                                                               y=self.robot_traj_pred["y_r"][i],
                                                                               psi=self.robot_traj_pred["psi_r"][i],
                                                                               width=self.car_width, length=self.car_length), closed=True, zorder=20, color="blue")
            robot_nopred_rect = matplotlib.patches.Polygon(
                self.polygon_xy_from_motionstate(x=self.robot_traj_nopred["x_r"][i],
                                                 y=self.robot_traj_nopred["y_r"][i],
                                                 psi=self.robot_traj_nopred["psi_r"][i],
                                                 width=self.car_width, length=self.car_length), closed=True, zorder=20, color="green")
            human_rect = matplotlib.patches.Polygon(self.polygon_xy_from_motionstate(x=self.human_traj["x_h"][i],
                                                                                     y=self.human_traj["y_h"][i],
                                                                                     psi=self.human_traj["psi_h"][i],
                                                                                     width=self.car_width,
                                                                                     length=self.car_length),
                                                    closed=True, zorder=20, color="red")
            axes.add_patch(robot_pred_rect)
            axes.add_patch(robot_nopred_rect)
            axes.add_patch(human_rect)

            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)

        plt.ioff()
        plt.show()

        return True

    def polygon_xy_from_motionstate(self, x, y, psi, width, length):
        lowleft = (x - length / 2., y - width / 2.)
        lowright = (x + length / 2., y - width / 2.)
        upright = (x + length / 2., y + width / 2.)
        upleft = (x - length / 2., y + width / 2.)
        return self.rotate_around_center(np.array([lowleft, lowright, upright, upleft]), np.array([x, y]),
                                    yaw=psi)

    def rotate_around_center(self, pts, center, yaw):
        return np.dot(pts - center, np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])) + center

    def load_data(self):

        # Map curbs
        self.intersection_curbs = np.load(
            "/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/intersection_curbs.npy")
        self.roundabout_curbs = np.load(
            "/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/roundabout_curbs.npy")

        # load car trajectory
        if self.scenario == "intersection":
            with open('/home/anjianl/Desktop/project/optimized_dp/result/plot_trajectory/intersection/2/pred_human.csv', 'rb') as f:
                self.human_traj = pickle.load(f)
            with open('/home/anjianl/Desktop/project/optimized_dp/result/plot_trajectory/intersection/2/pred_robot.csv', 'rb') as f:
                self.robot_traj_pred = pickle.load(f)
            with open('/home/anjianl/Desktop/project/optimized_dp/result/plot_trajectory/intersection/2/nopred_robot.csv',
                      'rb') as f:
                self.robot_traj_nopred = pickle.load(f)
        elif self.scenario == "roundabout":
            with open('/home/anjianl/Desktop/project/optimized_dp/result/plot_trajectory/roundabout/2/pred_human.csv', 'rb') as f:
                self.human_traj = pickle.load(f)
            with open('/home/anjianl/Desktop/project/optimized_dp/result/plot_trajectory/roundabout/2/pred_robot.csv', 'rb') as f:
                self.robot_traj_pred = pickle.load(f)
            with open('/home/anjianl/Desktop/project/optimized_dp/result/plot_trajectory/roundabout/2/nopred_robot.csv',
                      'rb') as f:
                self.robot_traj_nopred = pickle.load(f)


if __name__ == "__main__":
    PlotTrajectory().main()