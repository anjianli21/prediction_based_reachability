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

from prediction.clustering_v3 import ClusteringV3
from prediction.process_prediction_v3 import ProcessPredictionV3

import math

""" USER INTERFACES
- Define grid

- Generate initial values for grid using shape functions

- Time length for computations

- Run
"""
########################################################################################################################################
# Relative dynamics 5D
g = grid(np.array([-10.0, -10.0, -math.pi, -1, -1]), np.array([10.0, 10.0, math.pi, 18, 18]), 5, np.array([41, 41, 24, 39, 39]), [2])

# Define my object
# action_bound_mode = ClusteringV3().get_clustering()
# omega_bound, acc_bound = ProcessPredictionV3().omega_bound, ProcessPredictionV3().acc_bound

# mode: 0: decelerate, 1: stable, 2: accelerate, 3: left turn, 4: right turn, 5: in roundabout
my_car = RelDyn_5D(x=[0, 0, 0, 0, 0], uMin=np.array([-0.325, -5]), uMax=np.array([0.325, 3]),
                   dMin=np.array([-math.pi / 6, -5]), dMax=np.array([math.pi / 6, 5]), dims=5, uMode="max", dMode="min")


print("Computing relative dynamics 5D")

# Use the grid to initialize initial value function
Initial_value_f = CylinderShape(g, [3, 4, 5], np.zeros(5), 2)
Initial_value_f = np.minimum(Initial_value_f, HalfPlane(g, 0, 4))
Initial_value_f = np.minimum(Initial_value_f, - HalfPlane(g, 17, 4))

# Obstacles set is v_h \in [-1, 0] and [17, 18]
constraint_values = np.minimum(HalfPlane(g, 17, 3), - HalfPlane(g, 0, 3))

# Look-back length and time step
lookback_length = 3.1
t_step = 0.05

small_number = 1e-5
tau = np.arange(start = 0, stop = lookback_length + small_number, step = t_step)
print("Welcome to optimized_dp \n")

# Use the following variable to specify the characteristics of computation
# compMethod = "none" # Reachable set
compMethod = ["minVWithV0", "maxVWithCStraint"] # Reachable tube with obstacles

my_object  = my_car
my_shape = Initial_value_f
########################################################################################################################################

# # Bicycle dynamics 4D
# # Intersection scenario
# # Original 713 x 931 for intersection is too big
# # intersection_valfunc has size (366, 466, 24, 39)
# # roundabout_valfunc has size (367, 465, 24, 39)
# g = grid(np.array([940.8, 935.8, -math.pi, -1]), np.array([1066.7, 1035.1, math.pi, 18]), 4, np.array([366, 466, 24, 39]), [2])
#
# my_car = Bicycle_4D(x=[0, 0, 0, 0], uMin=np.array([-0.325, -5]), uMax=np.array([0.325, 3]), dims=4, uMode="max", dMode="min")
#
# print("Computing bicycle dynamics 4D")
#
# # Use the grid to initialize initial value function
# Initial_value_f = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/intersection_valfunc.npy") # Load the fmm map of the obstacle
# Initial_value_f = np.minimum(Initial_value_f, HalfPlane(g, 0, 3))
# Initial_value_f = np.minimum(Initial_value_f, - HalfPlane(g, 17, 3))
#
# # Look-back length and time step
# lookback_length = 3.1
# t_step = 0.05
#
# small_number = 1e-5
# tau = np.arange(start = 0, stop = lookback_length + small_number, step = t_step)
# print("Welcome to optimized_dp \n")
#
# # Use the following variable to specify the characteristics of computation
# # compMethod = "none" # Reachable set
# compMethod = ["minVWithV0", "maxVWithCStraint"] # Reachable tube with obstacles
#
# my_object  = my_car
# my_shape = Initial_value_f