from PIL import Image
import shutil

###################################################################################
# duplicate pngs
for i in range(1, 100):

    shutil.copy("/home/anjianl/Desktop/project/optimized_dp/result/speed/intersection/intersection_speed_profiles.png",
                "/home/anjianl/Desktop/project/optimized_dp/result/speed/intersection/intersection_speed_profiles_{:d}.png".format(i))


###################################################################################
to_gif = True

if to_gif:
    # Create the frames
    frames = []
    # imgs = glob.glob("*.png")
    imgs = []

    for i in range(1, 100):
        imgs.append(
            "/home/anjianl/Desktop/project/optimized_dp/result/speed/intersection/intersection_speed_profiles_{:d}.png".format(i))
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(
        "/home/anjianl/Desktop/project/optimized_dp/result/speed" + "/intersection_speed.gif",
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=300, loop=10)

    print("GIF is saved!")