import os
import cv2
import matplotlib.pyplot as plt 
import numpy as np 


import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pyelas', 'src'))

import elas


gastrec_path = "../kitti/input/2011_09_26/2011_09_26_drive_0001_sync"
ch_names = ["image_02", "image_03"]
fname = lambda x : "data/" + str(x).zfill(10) + ".png"

number = 50

# Images can be loaded with any library, as long as they can be accessed with buffer protocol
left_path = os.path.join(gastrec_path, ch_names[0], fname(number))
right_path = os.path.join(gastrec_path, ch_names[1], fname(number))
print(left_path)
print(right_path)
left = cv2.imread(left_path)
right = cv2.imread(right_path)
print(left.shape)

left_rgb = cv2.cvtColor(left,cv2.COLOR_BGR2RGB)
right_rgb = cv2.cvtColor(right,cv2.COLOR_BGR2RGB)


f, axarr = plt.subplots(2,2)
f.set_size_inches(30, 10)
axarr[0,0].imshow(left_rgb)
axarr[0,1].imshow(right_rgb)

###################
left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

disp_left = np.empty_like(left, dtype=np.float32)
disp_right = np.empty_like(right, dtype=np.float32)

params = elas.Elas_parameters()
params.postprocess_only_left = False
elas = elas.Elas(params)
elas.process_stereo(left, right, disp_left, disp_right)

scaled_depth_left = (disp_left / np.amax(disp_left) * 255.0).astype(np.uint8)

axarr[1,0].imshow(disp_left)
axarr[1,1].imshow(disp_right)

plt.savefig("my_tools/kitti.png")