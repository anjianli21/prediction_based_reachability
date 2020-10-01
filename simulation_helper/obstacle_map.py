from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

image = Image.open("/home/anjianl/Desktop/project/optimized_dp/data/map/image/intersection_curbs_color.png")
# image = Image.open("/home/anjianl/Desktop/project/optimized_dp/data/map/image/roundabout_curbs_color.png")


data = np.asarray(image)

print(data.shape)

image2 = Image.fromarray(data)

print(image2.size)

print(data[:, :, 0])

image3 = np.squeeze(data[:, :, 0])

image4 = np.zeros((np.shape(image3)))

for i in range(np.shape(image3)[0]):
    for j in range(np.shape(image3)[1]):
        if image3[i][j] > 220:
            image4[i][j] = 0
        else:
            image4[i][j] = 1

plt.imshow(image4)
plt.show()
# plt.savefig("/home/anjianl/Desktop/project/optimized_dp/data/map/intersection_obstacle_map.png")
# plt.savefig("/home/anjianl/Desktop/project/optimized_dp/data/map/roundabout_obstacle_map.png")

# np.save("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/roundabout_obs_map.npy", image4)
np.save("/home/anjianl/Desktop/project/optimized_dp/data/map/obstacle_map/intersection_obs_map.npy", image4)