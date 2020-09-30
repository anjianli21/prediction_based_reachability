import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import skfmm

"""
intersection
xmin:  940.8 xmax:  1066.7
ymin:  935.8 ymax:  1035.1
size is  731 931
dx, dy is 0.17222982216142282 0.13584131326949378

roundabout
xmin:  956.7 xmax:  1073.4
ymin:  954.0 ymax:  1046.0
size is  734 929
dx, dy is 0.1589918256130791 0.12534059945504086
"""

intersection_obs = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/intersection_obs_map.npy")
intersection_curbs = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/intersection_curbs.npy")
intersection_x_min, intersection_x_max = 940.8, 1066.7
intersection_y_min, intersection_y_max = 935.8, 1035.1

# intersection_obs = intersection_obs[::5, ::4]
print("intersection")
print("xmin: ", intersection_x_min, "xmax: ", intersection_x_max)
print("ymin: ", intersection_y_min, "ymax: ", intersection_y_max)
print("size is ", np.shape(intersection_obs)[0], np.shape(intersection_obs)[1])
print("dx, dy is", (intersection_x_max - intersection_x_min) / np.shape(intersection_obs)[0], (intersection_y_max - intersection_y_min) / np.shape(intersection_obs)[0])

intersection_fmm_prepare = intersection_obs
intersection_fmm_prepare[intersection_obs == 0] = - 1
intersection_fmm_prepare[intersection_obs == 1] = 1
intersection_fmm_map = skfmm.distance(intersection_fmm_prepare, dx=[0.17222982216142282, 0.13584131326949378])
intersection_fmm_map = - intersection_fmm_map
np.save("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/intersection_fmm_map.npy", intersection_fmm_map)

roundabout_obs = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/roundabout_obs_map.npy")
roundabout_curbs = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/roundabout_curbs.npy")
roundabout_x_min, roundabout_x_max = 956.7, 1073.4
roundabout_y_min, roundabout_y_max = 954.0, 1046.0
print("roundabout")
print("xmin: ", roundabout_x_min, "xmax: ", roundabout_x_max)
print("ymin: ", roundabout_y_min, "ymax: ", roundabout_y_max)
print("size is ", np.shape(roundabout_obs)[0], np.shape(roundabout_obs)[1])
print("dx, dy is", (roundabout_x_max - roundabout_x_min) / np.shape(roundabout_obs)[0], (roundabout_y_max - roundabout_y_min) / np.shape(roundabout_obs)[0])

roundabout_fmm_prepare = roundabout_obs
roundabout_fmm_prepare[roundabout_obs == 0] = - 1
roundabout_fmm_prepare[roundabout_obs == 1] = 1
roundabout_fmm_map = skfmm.distance(roundabout_obs, dx=[0.1589918256130791, 0.12534059945504086])
roundabout_fmm_map = - roundabout_fmm_map
np.save("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/roundabout_fmm_map.npy", roundabout_fmm_map)


# plt.contour(roundabout_fmm_map, levels=[0])
# plt.contour(roundabout_obs, levels=[0])

# plt.imshow(intersection_fmm_map)
# plt.show()

# plt.imshow(roundabout_fmm_map)
# plt.show()