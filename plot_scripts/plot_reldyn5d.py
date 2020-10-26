import sys
sys.path.append("/Users/anjianli/Desktop/robotics/project/optimized_dp")

import numpy as np
from Grid.GridProcessing import grid
from Shapes.ShapesFunctions import *

import math

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

class PlotReldyn5D(object):

    def plot(self):
        # g = grid(np.array([-10.0, -10.0, -math.pi, 0, 0]), np.array([10.0, 10.0, math.pi, 17, 17]), 5,
        #          np.array([41, 41, 36, 35, 35]), [2])
        # g = grid(np.array([-10.0, -10.0, -math.pi, -1, -1]), np.array([10.0, 10.0, math.pi, 18, 18]), 5,
        #          np.array([41, 41, 24, 39, 39]), [2])
        g = grid(np.array([-15.0, -10.0, -math.pi, -1, -1]), np.array([15.0, 10.0, math.pi, 18, 18]), 5,
                 np.array([61, 41, 24, 39, 39]), [2])


        # Set time
        time = 3.0
        # Previous
        # Set the slice parameter, not psi_rel = psi_h - psi_r
        psi = 45
        # v_h = 6.0
        v_h = 7.0
        v_r = 2.0
        # v_r = 1.0

        V_0 = np.load(
            "/home/anjianl/Desktop/project/optimized_dp/data/brs/1006/mode-1/reldyn5d_brs_mode-1_t_3.00.npy")
        V_1 = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/1006/mode1/reldyn5d_brs_mode1_t_3.00.npy")
        V_2 = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/1006/mode2/reldyn5d_brs_mode2_t_3.00.npy")
        V_3 = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/1006/mode3/reldyn5d_brs_mode3_t_3.00.npy")
        V_4 = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/1006/mode4/reldyn5d_brs_mode4_t_3.00.npy")
        V_5 = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/1006/mode5/reldyn5d_brs_mode5_t_3.00.npy")
        V_6 = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/1006/mode-1/reldyn5d_brs_mode-1_t_3.00.npy")

        x_grid, y_grid = self.get_xy_grid(g, [0, 1])

        psi_index = int((psi + 180) / 15)
        v_h_index = int(2 * v_h + 2)
        v_r_index = int(2 * v_r + 2)

        val_0 = np.squeeze(V_0[:, :, psi_index, v_h_index, v_r_index])
        val_1 = np.squeeze(V_1[:, :, psi_index, v_h_index, v_r_index])
        val_2 = np.squeeze(V_2[:, :, psi_index, v_h_index, v_r_index])
        val_3 = np.squeeze(V_3[:, :, psi_index, v_h_index, v_r_index])
        val_4 = np.squeeze(V_4[:, :, psi_index, v_h_index, v_r_index])
        val_5 = np.squeeze(V_5[:, :, psi_index, v_h_index, v_r_index])
        val_6 = np.squeeze(V_6[:, :, psi_index, v_h_index, v_r_index])


        fig, ax = plt.subplots()

        CS_0 = ax.contour(x_grid, y_grid, val_0, levels=[0], colors='sandybrown')
        ax.clabel(CS_0, inline=1, fontsize=5)
        CS_1 = ax.contour(x_grid, y_grid, val_1, levels=[0], colors='darkmagenta')
        ax.clabel(CS_1, inline=1, fontsize=5)
        CS_2 = ax.contour(x_grid, y_grid, val_2, levels=[0], colors='limegreen')
        ax.clabel(CS_2, inline=1, fontsize=5)
        CS_3 = ax.contour(x_grid, y_grid, val_3, levels=[0], colors='steelblue')
        ax.clabel(CS_3, inline=1, fontsize=5)
        CS_4 = ax.contour(x_grid, y_grid, val_4, levels=[0], colors='pink')
        ax.clabel(CS_4, inline=1, fontsize=5)
        CS_5 = ax.contour(x_grid, y_grid, val_5, levels=[0], colors='gold')
        ax.clabel(CS_5, inline=1, fontsize=5)
        CS_6 = ax.contour(x_grid, y_grid, val_6, levels=[0.0], colors='red')
        ax.clabel(CS_6, inline=1, fontsize=5)

        # lines = [CS_1.collections[0], CS_2.collections[0], CS_3.collections[0], CS_4.collections[0]]
        # labels = ['Mode 1: stable', 'Mode 2: acceleration', 'Mode 3: left turn', 'Mode 4: right turn']

        # lines = [CS_5.collections[0], CS_6.collections[0]]
        # labels = ['test1', "test2"]

        lines = [CS_0.collections[0], CS_1.collections[0], CS_2.collections[0], CS_3.collections[0], CS_4.collections[0], CS_5.collections[0], CS_6.collections[0]]
        labels = ['Mode 0: Deceleration', 'Mode 1: Stable', 'Mode2: Acceleration', 'Mode 3: Left turn', 'Mode 4: Right turn', 'Mode 5: Roundabout', "Mode -1: Others"]

        ax.legend(lines, labels)

        ax.set_xlabel("x_relative (m)")
        ax.set_ylabel("y_relative (m)")
        ax.set_title('BRT: psi_rel = {:.1f} rad, v_human = {:.1f} m/s, v_robot = {:.1f} m/s'.format(psi, v_h, v_r))


        # Plot control
        # CS_0 = ax.contour(x_grid, y_grid, ctrl_acc_val_0, levels=[-5, 3], colors='brown')
        # ax.clabel(CS_0, inline=1, fontsize=5)
        # ax.set_title('control acc')
        # CS_0 = ax.contour(x_grid, y_grid, ctrl_beta_val_0, levels=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3], colors='magenta')
        # ax.clabel(CS_0, inline=1, fontsize=5)
        # ax.set_title('control beta')


        plt.show()

    def get_xy_grid(self, grid, dims_plot):

        if len(dims_plot) != 2:
            raise Exception('dims_plot length should be equal to 2\n')
        else:
            dim1, dim2 = dims_plot[0], dims_plot[1]
            complex_x = complex(0, grid.pts_each_dim[dim1])
            complex_y = complex(0, grid.pts_each_dim[dim2])
            mg_X, mg_Y = np.mgrid[grid.min[dim1]:grid.max[dim1]: complex_x,
                               grid.min[dim2]:grid.max[dim2]: complex_y]

        return mg_X, mg_Y


if __name__ == '__main__':
    PlotReldyn5D().plot()