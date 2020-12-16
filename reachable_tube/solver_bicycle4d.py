import heterocl as hcl
import numpy as np
import time
#import plotly.graph_objects as go
import os

from reachable_tube.computeGraphs.CustomGraphFunctions import *
from reachable_tube.Plots.plotting_utilities import *
from reachable_tube.user_definer_bicycle4d import *
from argparse import ArgumentParser
from reachable_tube.computeGraphs.graph_4D import *
from reachable_tube.computeGraphs.graph_5D import *
from reachable_tube.computeGraphs.graph_6D import *
import scipy.io as sio

import matplotlib.pyplot as plt

import math


def main():
    ################### PARSING ARGUMENTS FROM USERS #####################

    parser = ArgumentParser()
    parser.add_argument("-p", "--plot", default=True, type=bool)
    # Print out LLVM option only
    parser.add_argument("-l", "--llvm", default=False, type=bool)
    args = parser.parse_args()

    hcl.init()
    hcl.config.init_dtype = hcl.Float()

    ################# INITIALIZE DATA TO BE INPUT INTO EXECUTABLE ##########################

    print("Initializing\n")

    V_0 = hcl.asarray(my_shape)
    V_1 = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))
    l0  = hcl.asarray(my_shape)
    #probe = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))
    #obstacle = hcl.asarray(cstraint_values)

    # Initialize uOpt_1, uOpt_2
    uOpt_1 = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))
    uOpt_2 = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))

    list_x1 = np.reshape(g.vs[0], g.pts_each_dim[0])
    list_x2 = np.reshape(g.vs[1], g.pts_each_dim[1])
    list_x3 = np.reshape(g.vs[2], g.pts_each_dim[2])
    if g.dims >= 4:
        list_x4 = np.reshape(g.vs[3], g.pts_each_dim[3])
    if g.dims >= 5:
        list_x5 = np.reshape(g.vs[4], g.pts_each_dim[4])
    if g.dims >= 6:
        list_x6 = np.reshape(g.vs[5], g.pts_each_dim[5])


    # Convert to hcl array type
    list_x1 = hcl.asarray(list_x1)
    list_x2 = hcl.asarray(list_x2)
    list_x3 = hcl.asarray(list_x3)
    if g.dims >= 4:
        list_x4 = hcl.asarray(list_x4)
    if g.dims >= 5:
        list_x5 = hcl.asarray(list_x5)
    if g.dims >= 6:
        list_x6 = hcl.asarray(list_x6)

    # Get executable
    if g.dims == 4:
        solve_pde = graph_4D()
    if g.dims == 5:
        solve_pde = graph_5D()
    if g.dims == 6:
        solve_pde = graph_6D()

    # Print out code for different backend
    print(solve_pde)

    ################ USE THE EXECUTABLE ############
    # Variables used for timing
    execution_time = 0
    lookback_time = 0

    tNow = tau[0]
    for i in range (1, len(tau)):
        #tNow = tau[i-1]
        t_minh= hcl.asarray(np.array((tNow, tau[i])))
        V1_old = V_1.asnumpy()
        while tNow <= tau[i] - 1e-4:
             # Start timing
             start = time.time()

             print("Started running\n")

             # Run the execution and pass input into graph
             if g.dims == 4:
                solve_pde(V_1, V_0, list_x1, list_x2, list_x3, list_x4, t_minh, l0, uOpt_1, uOpt_2)
             if g.dims == 5:
                solve_pde(V_1, V_0, list_x1, list_x2, list_x3, list_x4, list_x5, t_minh, l0, uOpt_1, uOpt_2)
             if g.dims == 6:
                solve_pde(V_1, V_0, list_x1, list_x2, list_x3, list_x4, list_x5, list_x6, t_minh, l0)

             # tNow = np.asscalar((t_minh.asnumpy())[0])
             tNow = (t_minh.asnumpy())[0].item() # Above line is deprecated since NumPy v1.16

             # Calculate computation time
             execution_time += time.time() - start

             # Some information printing
             print(t_minh)
             print("Computational time to integrate (s): {:.5f}".format(time.time() - start))

             print("tNow is ", tNow)

             # Saving reachable set and control data into disk
             if tNow == 0.00482916459441185 or tNow == 3:
                print("Max diff. from previous time step: {}".format(np.max(np.abs(V_1.asnumpy() - V1_old))))
                print("Avg diff. from previous time step: {}".format(np.mean(np.abs(V_1.asnumpy() - V1_old))))

                # TODO: change scenario here
                # # Intersection ###############################################################################
                # file_dir = '/home/anjianl/Desktop/project/optimized_dp/data/brs/1006/obstacle_buffer_1m/intersection'
                # if not os.path.exists(file_dir):
                #     os.mkdir(file_dir)
                # file_brs_path = file_dir + '/bicycle4d_brs_intersection' + '_t_%.2f.npy'
                # np.save(file_brs_path % tNow, V_1.asnumpy())
                # print("intersection brs is saved!")
                #
                # # Save control data
                # file_ctrl_beta_path = file_dir + '/bicycle4d_intersection_ctrl_beta' + '_t_%.2f.npy'
                # file_ctrl_acc_path = file_dir + '/bicycle4d_intersection_ctrl_acc' + '_t_%.2f.npy'
                # np.save(file_ctrl_beta_path % tNow, uOpt_1.asnumpy())
                # print("intersection control beta is saved!")
                # np.save(file_ctrl_acc_path % tNow, uOpt_2.asnumpy())
                # print("intersection control acc is saved!")

                # # Roundabout ##################################################################################
                file_dir = '/home/anjianl/Desktop/project/optimized_dp/data/brs/1006/obstacle_buffer_1m/roundabout'
                if not os.path.exists(file_dir):
                    os.mkdir(file_dir)
                file_brs_path = file_dir + '/bicycle4d_brs_roundabout' + '_t_%.2f.npy'
                np.save(file_brs_path % tNow, V_1.asnumpy())
                print("roundabout brs is saved!")

                # Save control data
                file_ctrl_beta_path = file_dir + '/bicycle4d_roundabout_ctrl_beta' + '_t_%.2f.npy'
                file_ctrl_acc_path = file_dir + '/bicycle4d_roundabout_ctrl_acc' + '_t_%.2f.npy'
                np.save(file_ctrl_beta_path % tNow, uOpt_1.asnumpy())
                print("roundabout control beta is saved!")
                np.save(file_ctrl_acc_path % tNow, uOpt_2.asnumpy())
                print("roundabout control acc is saved!")

        print("Max diff. from previous time step: {}".format(np.max(np.abs(V_1.asnumpy() - V1_old))))
        print("Avg diff. from previous time step: {}".format(np.mean(np.abs(V_1.asnumpy() - V1_old))))

    # Time info printing
    print("Total kernel time (s): {:.5f}".format(execution_time))
    print("Finished solving\n")

    # V1 is the final value array, fill in anything to use it



    ##################### PLOTTING #####################
    # if args.plot:
    #     plot_isosurface(g, V_1.asnumpy(), [0, 1, 3])


if __name__ == '__main__':

    main()
