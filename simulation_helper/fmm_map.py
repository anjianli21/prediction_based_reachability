import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import skfmm

"""
intersection
xmin:  940.8 xmax:  1066.7
ymin:  935.8 ymax:  1035.1
size is  931 * 731 (x * y)
dx, dy is 0.13523093447905488 0.13584131326949378
intersection fmm downsampled size is (466, 366)
"""

intersection_obs = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/intersection_obs_map.npy").T
# intersection_curbs = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/intersection_curbs.npy")
intersection_x_min, intersection_x_max = 940.8, 1066.7
intersection_y_min, intersection_y_max = 935.8, 1035.1
intersection_dx, intersection_dy = (intersection_x_max - intersection_x_min) / np.shape(intersection_obs)[0], (intersection_y_max - intersection_y_min) / np.shape(intersection_obs)[1]

print("intersection")
print("xmin: ", intersection_x_min, "xmax: ", intersection_x_max)
print("ymin: ", intersection_y_min, "ymax: ", intersection_y_max)
print("size is ", np.shape(intersection_obs)[0], np.shape(intersection_obs)[1])
print("dx, dy is", intersection_dx, intersection_dy)

# Compute fmm distance
intersection_fmm_prepare = intersection_obs
intersection_fmm_prepare[intersection_obs == 0] = - 1
intersection_fmm_prepare[intersection_obs == 1] = 1
intersection_fmm_map = skfmm.distance(intersection_fmm_prepare, dx=[intersection_dx, intersection_dy])
intersection_fmm_map = - intersection_fmm_map
# Add buffer area for obstacle map, margin = 1.5m
intersection_obs_buffer = np.zeros(np.shape(intersection_fmm_map))
intersection_obs_buffer[intersection_fmm_map <= 1.5] = 1
intersection_obs_buffer[intersection_fmm_map > 1.5] = -1
intersection_buffer_fmm_map = skfmm.distance(intersection_obs_buffer, dx=[intersection_dx, intersection_dy])
intersection_buffer_fmm_map = - intersection_buffer_fmm_map

#############################
# # Without buffer area
# np.save("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/intersection_fmm_map.npy", intersection_fmm_map)
#
# intersection_fmm_map_downsampled = intersection_fmm_map[::2, ::2]
# print("intersection fmm downsampled size is", np.shape(intersection_fmm_map_downsampled))
# np.save("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/intersection_fmm_map_downsampled.npy", intersection_fmm_map_downsampled)
#
# intersection_valfunc = np.zeros((466, 366, 24, 39))
# for i in range(24):
#     for j in range(39):
#         intersection_valfunc[:, :, i, j] = intersection_fmm_map_downsampled
#
# # TODO, reverse y axis, I don't know why
# intersection_valfunc_correctify = np.zeros(np.shape(intersection_valfunc))
# for i in range(366):
#     intersection_valfunc_correctify[:, i, :, :] = intersection_valfunc[:, 365 - i, :, :]
#
# np.save("/home/anjianl/Desktop/project/optimized_dp/data/map/value_function/intersection_valfunc.npy", intersection_valfunc)
# np.save("/home/anjianl/Desktop/project/optimized_dp/data/map/value_function/intersection_valfunc_correct.npy", intersection_valfunc_correctify)

#############################
# # With buffer area
# np.save("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/intersection_fmm_map_buffer_1.5.npy", intersection_buffer_fmm_map)
#
# intersection_buffer_fmm_map_downsampled = intersection_buffer_fmm_map[::2, ::2]
# print("intersection fmm buffer downsampled size is", np.shape(intersection_buffer_fmm_map_downsampled))
# # np.save("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/intersection_fmm_map_downsampled_buffer_1.5.npy", intersection_buffer_fmm_map_downsampled)
#
# intersection_valfunc_buffer = np.zeros((466, 366, 24, 39))
# for i in range(24):
#     for j in range(39):
#         intersection_valfunc_buffer[:, :, i, j] = intersection_buffer_fmm_map_downsampled
#
# # TODO, reverse y axis, I don't know why
# intersection_valfunc_buffer_correctify = np.zeros(np.shape(intersection_valfunc_buffer))
# for i in range(366):
#     intersection_valfunc_buffer_correctify[:, i, :, :] = intersection_valfunc_buffer[:, 365 - i, :, :]
#
# np.save("/home/anjianl/Desktop/project/optimized_dp/data/map/value_function/intersection_valfunc_correct_buffer_1.5.npy", intersection_valfunc_buffer_correctify)

# plt.imshow(intersection_fmm_map_downsampled.T)
# plt.show()

##################################################################################################################################

# """
# roundabout
# xmin:  956.7 xmax:  1073.4
# ymin:  954.0 ymax:  1046.0
# size is  929 * 734 (x * y)
# dx, dy is 0.12561894510226054 0.12534059945504086
# roundabout fmm downsampled size is (465, 367)
# """

roundabout_obs = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/roundabout_obs_map.npy").T
# roundabout_curbs = np.load("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/roundabout_curbs.npy")
roundabout_x_min, roundabout_x_max = 956.7, 1073.4
roundabout_y_min, roundabout_y_max = 954.0, 1046.0
roundabout_dx, roundabout_dy = (roundabout_x_max - roundabout_x_min) / np.shape(roundabout_obs)[0], (roundabout_y_max - roundabout_y_min) / np.shape(roundabout_obs)[1]

print("roundabout")
print("xmin: ", roundabout_x_min, "xmax: ", roundabout_x_max)
print("ymin: ", roundabout_y_min, "ymax: ", roundabout_y_max)
print("size is ", np.shape(roundabout_obs)[0], np.shape(roundabout_obs)[1])
print("dx, dy is", roundabout_dx, roundabout_dy)

roundabout_fmm_prepare = roundabout_obs
roundabout_fmm_prepare[roundabout_obs == 0] = - 1
roundabout_fmm_prepare[roundabout_obs == 1] = 1
roundabout_fmm_map = skfmm.distance(roundabout_fmm_prepare, dx=[roundabout_dx, roundabout_dy])
roundabout_fmm_map = - roundabout_fmm_map
# np.save("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/roundabout_fmm_map.npy", roundabout_fmm_map)
# Add buffer area for obstacle map, margin = 1.5m
roundabout_obs_buffer = np.zeros(np.shape(roundabout_fmm_map))
roundabout_obs_buffer[roundabout_fmm_map <= 1.5] = 1
roundabout_obs_buffer[roundabout_fmm_map > 1.5] = -1
roundabout_buffer_fmm_map = skfmm.distance(roundabout_obs_buffer, dx=[roundabout_dx, roundabout_dy])
roundabout_buffer_fmm_map = - roundabout_buffer_fmm_map

#############################
# # Without buffer area
# roundabout_fmm_map_downsampled = roundabout_fmm_map[::2, ::2]
# print("roundabout fmm downsampled size is", np.shape(roundabout_fmm_map_downsampled))
# np.save("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/roundabout_fmm_map_downsampled.npy", roundabout_fmm_map_downsampled)
#
# roundabout_valfunc = np.zeros((465, 367, 24, 39))
# for i in range(24):
#     for j in range(39):
#         roundabout_valfunc[:, :, i, j] = roundabout_fmm_map_downsampled
#
# # TODO, reverse y axis, I don't know why
# roundabout_valfunc_correctify = np.zeros(np.shape(roundabout_valfunc))
# for i in range(367):
#     roundabout_valfunc_correctify[:, i, :, :] = roundabout_valfunc[:, 366 - i, :, :]
#
# np.save("/home/anjianl/Desktop/project/optimized_dp/data/map/value_function/roundabout_valfunc.npy", roundabout_valfunc)
# np.save("/home/anjianl/Desktop/project/optimized_dp/data/map/value_function/roundabout_valfunc_correct.npy", roundabout_valfunc_correctify)

#############################
# With buffer area
np.save("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/roundabout_fmm_map_buffer_1.5.npy", roundabout_buffer_fmm_map)

roundabout_buffer_fmm_map_downsampled = roundabout_buffer_fmm_map[::2, ::2]
print("roundabout fmm buffer downsampled size is", np.shape(roundabout_buffer_fmm_map_downsampled))
# np.save("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/roundabout_fmm_map_downsampled_buffer_1.5.npy", roundabout_buffer_fmm_map_downsampled)

roundabout_valfunc_buffer = np.zeros((465, 367, 24, 39))
for i in range(24):
    for j in range(39):
        roundabout_valfunc_buffer[:, :, i, j] = roundabout_buffer_fmm_map_downsampled

# TODO, reverse y axis, I don't know why
roundabout_valfunc_buffer_correctify = np.zeros(np.shape(roundabout_valfunc_buffer))
for i in range(366):
    roundabout_valfunc_buffer_correctify[:, i, :, :] = roundabout_valfunc_buffer[:, 366 - i, :, :]

np.save("/home/anjianl/Desktop/project/optimized_dp/data/map/value_function/roundabout_valfunc_correct_buffer_1.5.npy", roundabout_valfunc_buffer_correctify)

#
# plt.imshow(roundabout_fmm_map_downsampled.T)
# plt.show()