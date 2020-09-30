import numpy as np

print(range(6))

for i in range(-1, 6, 1):
    print(i)
    mode_0_brs = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/0928/mode{:d}/reldyn5d_brs_mode{:d}_t_3.00.npy".format(i, i))

    mode_0_brs_correct = np.append(mode_0_brs, mode_0_brs[:, :, 0, None, :, :], axis=2)

    np.save("/home/anjianl/Desktop/project/optimized_dp/data/brs/0928-correct/mode{:d}/reldyn5d_brs_mode{:d}_t_3.00.npy".format(i, i), mode_0_brs_correct)

    mode_0_acc = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/0928/mode{:d}/reldyn5d_ctrl_acc_mode{:d}_t_3.00.npy".format(i, i))

    mode_0_acc_correct = np.append(mode_0_acc, mode_0_acc[:, :, 0, None, :, :], axis=2)

    np.save("/home/anjianl/Desktop/project/optimized_dp/data/brs/0928-correct/mode{:d}/reldyn5d_ctrl_acc_mode{:d}_t_3.00.npy".format(i, i), mode_0_acc_correct)

    mode_0_beta = np.load("/home/anjianl/Desktop/project/optimized_dp/data/brs/0928/mode{:d}/reldyn5d_ctrl_beta_mode{:d}_t_3.00.npy".format(i, i))

    mode_0_beta_correct = np.append(mode_0_beta, mode_0_beta[:, :, 0, None, :, :], axis=2)

    np.save("/home/anjianl/Desktop/project/optimized_dp/data/brs/0928-correct/mode{:d}/reldyn5d_ctrl_beta_mode{:d}_t_3.00.npy".format(i, i), mode_0_beta_correct)