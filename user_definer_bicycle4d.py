import numpy as np
from Grid.GridProcessing import grid
from Shapes.ShapesFunctions import *

# Specify the  file that includes dynamic systems
from dynamics.Humannoid6D_sys1 import *
from dynamics.DubinsCar4D import *
from dynamics.DubinsCar import *
from dynamics.RelDyn5D import *
from dynamics.Bicycle4D import *
import scipy.io as sio
from matplotlib import pyplot as plt

from prediction.clustering import Clustering
from prediction.process_prediction import ProcessPrediction

import math

""" USER INTERFACES
- Define grid

- Generate initial values for grid using shape functions

- Time length for computations

- Run
"""

########################################################################################################################################

# Bicycle dynamics 4D
# Intersection scenario
# xmin:  940.8 xmax:  1066.7
# ymin:  935.8 ymax:  1035.1
# intersection fmm downsampled size is (466, 366)

# g = grid(np.array([940.8, 935.8, -math.pi, -1]), np.array([1066.7, 1035.1, math.pi, 18]), 4, np.array([466, 366, 24, 39]), [2])
#
# my_car = Bicycle_4D(x=[0, 0, 0, 0], uMin=np.array([-0.325, -5]), uMax=np.array([0.325, 3]), dims=4, uMode="max", dMode="min")
#
# print("Computing bicycle dynamics 4D intersection")
#
# # Use the grid to initialize initial value function
# # Initial_value_f = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/value_function/intersection_valfunc_correct.npy") # Load the fmm map-based value function of the obstacle
# # Use value function with buffer = 1.5m
# # Initial_value_f = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/value_function/intersection_valfunc_correct_buffer_1.5.npy")
# # Use value function with buffer = 1m
# Initial_value_f = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/value_function/intersection_valfunc_correct_buffer_1m.npy")
# # Add speed target
# Initial_value_f = np.minimum(Initial_value_f, HalfPlane(g, 0, 3))
# Initial_value_f = np.minimum(Initial_value_f, - HalfPlane(g, 17, 3))
#
# # Look-back length and time step
# lookback_length = 4.1
# t_step = 0.05
#
# small_number = 1e-5
# tau = np.arange(start = 0, stop = lookback_length + small_number, step = t_step)
# print("Welcome to optimized_dp \n")
#
# # Use the following variable to specify the characteristics of computation
# # compMethod = "none" # Reachable set
# compMethod = "minVWithV0" # Reachable tube
#
# my_object  = my_car
# my_shape = Initial_value_f

########################################################################################################################################

# Bicycle dynamics 4D
# Roundabout scenario
# xmin:  956.7 xmax:  1073.4
# ymin:  954.0 ymax:  1046.0
# roundabout fmm downsampled size is (465, 367)

g = grid(np.array([956.7, 954.0, -math.pi, -1]), np.array([1073.4, 1046.0, math.pi, 18]), 4, np.array([465, 367, 24, 39]), [2])

my_car = Bicycle_4D(x=[0, 0, 0, 0], uMin=np.array([-0.325, -5]), uMax=np.array([0.325, 3]), dims=4, uMode="max", dMode="min")

print("Computing bicycle dynamics 4D roundabout")

# Use the grid to initialize initial value function
# Initial_value_f = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/value_function/roundabout_valfunc_correct.npy") # Load the fmm map-based value function of the obstacle
# Use value function with buffer = 1.5m
# Initial_value_f = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/value_function/roundabout_valfunc_correct_buffer_1.5.npy")
# Use buffer = 1m
Initial_value_f = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/value_function/roundabout_valfunc_correct_buffer_1m.npy")
# Add speed target
Initial_value_f = np.minimum(Initial_value_f, HalfPlane(g, 0, 3))
Initial_value_f = np.minimum(Initial_value_f, - HalfPlane(g, 17, 3))

# Look-back length and time step
lookback_length = 3.1
t_step = 0.05

small_number = 1e-5
tau = np.arange(start = 0, stop = lookback_length + small_number, step = t_step)
print("Welcome to optimized_dp \n")

# Use the following variable to specify the characteristics of computation
# compMethod = "none" # Reachable set
compMethod = "minVWithV0" # Reachable tube

my_object  = my_car
my_shape = Initial_value_f