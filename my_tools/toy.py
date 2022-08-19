import tifffile
import numpy as np 
import matplotlib.pyplot as plt

path = "/raid/sghong/DepthEstimation/scared/dataset_1/keyframe_3/left_depth_map.tiff"

import tifffile as tiff
a = tiff.imread(path)[:,:,-1]


a = np.round(a)
a[np.isnan(a)] = 0
a = a.astype('uint8')

print(a)
plt.imshow(a)
plt.imsave("aa.png", a)