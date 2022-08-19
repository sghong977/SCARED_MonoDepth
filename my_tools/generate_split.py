import os

in_path = "" #"/raid/datasets/hutom/gastric/os_frames"
out_path = "splits/gast_test.txt"

R = "R000007"
chs = "ch1_video_04/"  #os.listdir(os.path.join(in_path, R))
tail = "518.8579"

with open(out_path, "w") as f:
    fname = lambda x : "frame" + str(x).zfill(10) + ".jpg"
    for i in range(1000,2000):
        img = os.path.join(in_path, R, chs, fname(i))
        png = os.path.join(in_path, R, chs, fname(i))
        f.write(img + " " + png + " " + tail + "\n")