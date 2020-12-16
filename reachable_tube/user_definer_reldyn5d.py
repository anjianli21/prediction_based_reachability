import numpy as np
from reachable_tube.Grid.GridProcessing import grid
from reachable_tube.Shapes.ShapesFunctions import *

# Specify the  file that includes dynamic systems
from reachable_tube.dynamics.Humannoid6D_sys1 import *
from reachable_tube.dynamics.DubinsCar4D import *
from reachable_tube.dynamics.DubinsCar import *
from reachable_tube.dynamics.RelDyn5D import *
from reachable_tube.dynamics.Bicycle4D import *
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
# Relative dynamics 5D
# g = grid(np.array([-10.0, -10.0, -math.pi, -1, -1]), np.array([10.0, 10.0, math.pi, 18, 18]), 5, np.array([41, 41, 24, 39, 39]), [2])
g = grid(np.array([-15.0, -10.0, -math.pi, -1, -1]), np.array([15.0, 10.0, math.pi, 18, 18]), 5, np.array([61, 41, 24, 39, 39]), [2])

# Define my object
# action_bound_mode = ClusteringV3().get_clustering()
# omega_bound, acc_bound = ProcessPredictionV3().omega_bound, ProcessPredictionV3().acc_bound

# mode: 0: decelerate, 1: stable, 2: accelerate, 3: left turn, 4: right turn, 5: in roundabout
my_car = RelDyn_5D(x=[0, 0, 0, 0, 0], uMin=np.array([-0.325, -5]), uMax=np.array([0.325, 3]),
                   dMin=np.array([-math.pi / 6, -5]), dMax=np.array([math.pi / 6, 5]), dims=5, uMode="max", dMode="min")


print("Computing relative dynamics 5D")

# Use the grid to initialize initial value function
# TODO: previously, the radius is 2
# Initial_value_f = CylinderShape(g, [3, 4, 5], np.zeros(5), 2)
# TODO: Currently, we try a rectangle, with x has radius of 4, y has radius of 3
# Initial_value_f = RectangleXY(g, x_radius=4, y_radius=3)
# TODO 4x3 might be too big, we use 3.5 x 1.5
Initial_value_f = RectangleXY(g, x_radius=3.5, y_radius=1.5)
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