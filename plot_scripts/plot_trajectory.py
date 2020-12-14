import sys
sys.path.append("/home/anjianl/Desktop/project/optimized_dp")

from plot_utils import map_vis_without_lanelet

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle
import argparse

from PIL import Image

from matplotlib.widgets import Button

from plot_utils import dataset_reader
from plot_utils import dataset_types
from plot_utils import map_vis_lanelet2
from plot_utils import tracks_vis
from plot_utils import dict_utils

class PlotTrajectory(object):

    def __init__(self):

        self.car_length = 2.8
        self.car_width = 1

        # self.save_plot = False
        self.save_plot = True

        self.save_gif = True

        # self.car_to_plot = ["pred"]
        self.car_to_plot = ["nopred"]

    def main(self):

        # Input scenario name
        parser = argparse.ArgumentParser()
        parser.add_argument("scenario_name", type=str,
                            help="Name of the scenario (to identify map and folder for track "
                                 "files)", nargs="?")
        args = parser.parse_args()

        self.scenario = args.scenario_name

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

        for i in range(len(self.robot_traj_pred["t"])):

            plt.cla()
            map_vis_without_lanelet.draw_map_without_lanelet(lanelet_map_file, axes, lat_origin, lon_origin)
            axes.plot(self.human_traj["x_h"][:i], self.human_traj["y_h"][:i], "-r", label="human trajectory", linewidth=0.5)
            if "pred" in self.car_to_plot:
                axes.plot(self.robot_traj_pred["x_r"][:i], self.robot_traj_pred["y_r"][:i], "-b", label="robot trajectory prediction", linewidth=0.5)  # with prediction
                robot_pred_rect = matplotlib.patches.Polygon(
                    self.polygon_xy_from_motionstate(x=self.robot_traj_pred["x_r"][i],
                                                     y=self.robot_traj_pred["y_r"][i],
                                                     psi=self.robot_traj_pred["psi_r"][i],
                                                     width=self.car_width, length=self.car_length), closed=True,
                    zorder=20, color="blue")
                axes.add_patch(robot_pred_rect)
            if "nopred" in self.car_to_plot:
                axes.plot(self.robot_traj_nopred["x_r"][:i], self.robot_traj_nopred["y_r"][:i], "-g", label="robot trajectory no prediction", linewidth=0.5)  # no prediction
                robot_nopred_rect = matplotlib.patches.Polygon(
                    self.polygon_xy_from_motionstate(x=self.robot_traj_nopred["x_r"][i],
                                                     y=self.robot_traj_nopred["y_r"][i],
                                                     psi=self.robot_traj_nopred["psi_r"][i],
                                                     width=self.car_width, length=self.car_length), closed=True,
                    zorder=20, color="green")
                axes.add_patch(robot_nopred_rect)
            # plt.plot(cx, cy, ".c", label="course")

            human_rect = matplotlib.patches.Polygon(self.polygon_xy_from_motionstate(x=self.human_traj["x_h"][i],
                                                                                     y=self.human_traj["y_h"][i],
                                                                                     psi=self.human_traj["psi_h"][i],
                                                                                     width=self.car_width,
                                                                                     length=self.car_length),
                                                    closed=True, zorder=20, color="red")
            axes.add_patch(human_rect)

            if self.car_to_plot == ["pred", "nopred"]:
                title = "time: {:.1f}s".format(self.robot_traj_pred["t"][i])
            elif "pred" in self.car_to_plot:
                title = "time: {:.1f}s".format(self.robot_traj_pred["t"][i]) + ", max deviation: {:.2f}m".format(self.robot_traj_pred["max_deviation"][i])
            elif "nopred" in self.car_to_plot:
                title = "time: {:.1f}s".format(self.robot_traj_pred["t"][i]) + ", max deviation: {:.2f}m".format(self.robot_traj_nopred["max_deviation"][i])

            fig.suptitle(title)
            axes.set_xlabel('x position (m)')
            axes.set_ylabel('y position (m)')
            axes.margins(-0.2, -0.2) # zoom in a bit

            plt.axis("equal")
            # plt.grid(True)
            plt.pause(0.001)

            if self.save_plot:
                if self.car_to_plot == ["pred", "nopred"]:
                    if self.scenario == "intersection":
                        folder_path = "/home/anjianl/Desktop/project/optimized_dp/result/plot_trajectory/intersection/plots3"
                    else:
                        folder_path = "/home/anjianl/Desktop/project/optimized_dp/result/plot_trajectory/roundabout/plots3"
                    plt.savefig(folder_path + "/3_cars/t_{:.2f}.png".format(self.human_traj["t"][i]))
                    print("trajectory shots are saved!")
                elif "pred" in self.car_to_plot:
                    if self.scenario == "intersection":
                        folder_path = "/home/anjianl/Desktop/project/optimized_dp/result/plot_trajectory/intersection/plots3"
                    else:
                        folder_path = "/home/anjianl/Desktop/project/optimized_dp/result/plot_trajectory/roundabout/plots3"
                    plt.savefig(folder_path + "/2_car_pred/t_{:.2f}.png".format(self.human_traj["t"][i]))
                    print("trajectory shots are saved!")
                elif "nopred" in self.car_to_plot:
                    if self.scenario == "intersection":
                        folder_path = "/home/anjianl/Desktop/project/optimized_dp/result/plot_trajectory/intersection/plots3"
                    else:
                        folder_path = "/home/anjianl/Desktop/project/optimized_dp/result/plot_trajectory/roundabout/plots3"
                    plt.savefig(folder_path + "/2_car_nopred/t_{:.2f}.png".format(self.human_traj["t"][i]))
                    print("trajectory shots are saved!")

        plt.subplots(1)
        plt.plot(self.robot_traj_pred["t"], self.robot_traj_pred["v_r"], label="Reachability-Pred (Ours)")
        plt.plot(self.robot_traj_nopred["t"], self.robot_traj_nopred["v_r"], label="Reachability-NoPred (Baseline)")
        plt.xlabel('time (s)')
        plt.ylabel('speed (m/s)')

        plt.legend()
        if self.save_plot:
            plt.savefig(folder_path + "/speed_profiles.png")
            print("speed profile is saved!")

        plt.ioff()
        # plt.show()

        if self.save_gif:
            if self.scenario == "intersection":
                tmp_folder_path = "/home/anjianl/Desktop/project/optimized_dp/result/plot_trajectory/intersection/plots3/"
                max_time = 9.9

                self.get_gif(tmp_folder_path, max_time, "pred")
                self.get_gif(tmp_folder_path, max_time, "nopred")
            elif self.scenario == "roundabout":
                tmp_folder_path = "/home/anjianl/Desktop/project/optimized_dp/result/plot_trajectory/roundabout/plots3/"
                max_time = 9.9

                self.get_gif(tmp_folder_path, max_time, "pred")
                self.get_gif(tmp_folder_path, max_time, "nopred")


        return True

    def get_gif(self, tmp_folder_path, max_t, pred_type):

        # Create the frames
        frames = []
        # imgs = glob.glob("*.png")
        imgs = []

        if pred_type == "pred":
            data_dir = tmp_folder_path + "2_car_pred/"
        elif pred_type == "nopred":
            data_dir = tmp_folder_path + "2_car_nopred/"

        for i in np.arange(0.00, max_t, 0.10):
            imgs.append(
                data_dir + "t_{:.2f}.png".format(i))
        for i in imgs:
            new_frame = Image.open(i)
            frames.append(new_frame)

        # Save into a GIF file that loops forever
        save_dir = tmp_folder_path
        if pred_type == "pred":
            file_name = "2_car_pred.gif"
        elif pred_type == "nopred":
            file_name = "2_car_nopred.gif"
        frames[0].save(
            save_dir + file_name,
            format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=300, loop=0)

        print("GIF is saved!")

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
            with open('/home/anjianl/Desktop/project/optimized_dp/result/plot_trajectory/intersection/1104/pred_human.csv', 'rb') as f:
                self.human_traj = pickle.load(f)
            with open('/home/anjianl/Desktop/project/optimized_dp/result/plot_trajectory/intersection/1104/pred_robot.csv', 'rb') as f:
                self.robot_traj_pred = pickle.load(f)
            with open('/home/anjianl/Desktop/project/optimized_dp/result/plot_trajectory/intersection/1104/nopred_robot.csv',
                      'rb') as f:
                self.robot_traj_nopred = pickle.load(f)
        elif self.scenario == "roundabout":
            with open('/home/anjianl/Desktop/project/optimized_dp/result/plot_trajectory/roundabout/1104/pred_human.csv', 'rb') as f:
                self.human_traj = pickle.load(f)
            with open('/home/anjianl/Desktop/project/optimized_dp/result/plot_trajectory/roundabout/1104/pred_robot.csv', 'rb') as f:
                self.robot_traj_pred = pickle.load(f)
            with open('/home/anjianl/Desktop/project/optimized_dp/result/plot_trajectory/roundabout/1104/nopred_robot.csv',
                      'rb') as f:
                self.robot_traj_nopred = pickle.load(f)


if __name__ == "__main__":
    PlotTrajectory().main()