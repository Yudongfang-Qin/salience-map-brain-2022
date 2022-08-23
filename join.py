import numpy as np
temp = np.load('saliency_map2_gaussian_3_500.npy')
print(np.shape(temp))

for num in np.arange(1000,5001,500):
    temp = np.concatenate((temp,np.load("saliency_map2_gaussian_3_"+str(num)+".npy")))
    print(np.shape(temp))
np.save('ukbb_saliency_map_female_gaussian_3.npy',temp)

