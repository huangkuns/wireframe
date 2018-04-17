import numpy as np
# 0 menas no rotation.
#rotation_angles = [i * 30. for i in range(12)] 
rotation_angles = [0, 180] 
# [0, 30, 60, 90, 120, 150, ..., 330]

crop_ration = [(0.25, 0.25, 0.75, 0.75)]

#pixel_mean = np.array([116.46479065, 126.95309567,  137.88588786], dtype=np.float32)

#pixel_mean = np.array([115.9839754, 126.63120922, 137.73309306], dtype=np.float32)
#pixel_mean = np.array([116.465, 126.953, 137.886], dtype=np.float32)
pixel_mean = np.array([116., 127., 138.], dtype=np.float32)
