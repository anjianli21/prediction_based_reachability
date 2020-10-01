import numpy as np

# # For relative system 5D
# for i in range(-1, 6, 1):
#     print(i)
#
#     file_dir = "/home/anjianl/Desktop/project/optimized_dp/data/brs/0929/"
#     file_dir_new = "/home/anjianl/Desktop/project/optimized_dp/data/brs/0929-correct/"
#
#     mode_0_brs = np.load(file_dir + "mode{:d}/reldyn5d_brs_mode{:d}_t_3.00.npy".format(i, i))
#
#     mode_0_brs_correct = np.append(mode_0_brs, mode_0_brs[:, :, 0, None, :, :], axis=2)
#
#     np.save(file_dir_new + "mode{:d}/reldyn5d_brs_mode{:d}_t_3.00.npy".format(i, i), mode_0_brs_correct)
#
#     mode_0_acc = np.load(file_dir + "mode{:d}/reldyn5d_ctrl_acc_mode{:d}_t_3.00.npy".format(i, i))
#
#     mode_0_acc_correct = np.append(mode_0_acc, mode_0_acc[:, :, 0, None, :, :], axis=2)
#
#     np.save(file_dir_new + "mode{:d}/reldyn5d_ctrl_acc_mode{:d}_t_3.00.npy".format(i, i), mode_0_acc_correct)
#
#     mode_0_beta = np.load(file_dir + "mode{:d}/reldyn5d_ctrl_beta_mode{:d}_t_3.00.npy".format(i, i))
#
#     mode_0_beta_correct = np.append(mode_0_beta, mode_0_beta[:, :, 0, None, :, :], axis=2)
#
#     np.save(file_dir_new + "mode{:d}/reldyn5d_ctrl_beta_mode{:d}_t_3.00.npy".format(i, i), mode_0_beta_correct)

##########################################################################################################################
# For 4D bicycle obstacles.
file_dir = "/home/anjianl/Desktop/project/optimized_dp/data/brs/0929/obstacle/intersection/"
file_dir_new = "/home/anjianl/Desktop/project/optimized_dp/data/brs/0929-correct/obstacle/intersection/"

# brs
intersection_brs = np.load(file_dir + "bicycle4d_brs_intersection_t_3.00.npy")
intersection_brs_correct = np.append(intersection_brs, intersection_brs[:, :, 0, None, :], axis=2)
np.save(file_dir_new + "bicycle4d_brs_intersection_t_3.00.npy", intersection_brs_correct)

# ctrl acc
intersection_ctrl_acc = np.load(file_dir + "bicycle4d_intersection_ctrl_acc_t_3.00.npy")
intersection_ctrl_acc_correct = np.append(intersection_ctrl_acc, intersection_ctrl_acc[:, :, 0, None, :], axis=2)
np.save(file_dir_new + "bicycle4d_intersection_ctrl_acc_t_3.00.npy", intersection_ctrl_acc_correct)

# ctrl beta
intersection_ctrl_beta = np.load(file_dir + "bicycle4d_intersection_ctrl_beta_t_3.00.npy")
intersection_ctrl_beta_correct = np.append(intersection_ctrl_beta, intersection_ctrl_beta[:, :, 0, None, :], axis=2)
np.save(file_dir_new + "bicycle4d_intersection_ctrl_beta_t_3.00.npy", intersection_ctrl_beta_correct)