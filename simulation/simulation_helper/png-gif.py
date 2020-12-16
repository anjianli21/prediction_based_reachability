from PIL import Image
import numpy as np
import glob

# Create the frames
frames = []
# imgs = glob.glob("*.png")
imgs = []
for i in np.arange(0.10, 6.80, 0.10):
    imgs.append("/home/anjianl/Desktop/project/optimized_dp/result/simulation/1006/roundabout/3/pred/t_{:.2f}.png".format(i))
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save('/home/anjianl/Desktop/project/optimized_dp/result/simulation/1006/roundabout/3/3_with_prediction.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=300, loop=0)