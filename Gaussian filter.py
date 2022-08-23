import numpy as np
from scipy import ndimage

data = np.load('ukbb_saliency_map_female.npy')


a=np.zeros(data.shape)
for i in range(len(data)):
    for j in range(data.shape[3]):
        a[i, :, :, j] = ndimage.gaussian_filter(data[i, :, :, j], sigma=(2, 2), order=0)
np.save("gaussian_filter_3.npy", a)
