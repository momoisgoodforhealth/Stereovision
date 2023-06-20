import numpy as np

K1 = np.zeros((3, 3))
D1 = np.zeros((4, 1))
K2 = np.zeros((3, 3))
D2 = np.zeros((4, 1))


print(K1.tolist()) # camera_matrix
print(D1.tolist()) # dist_coefs
print(K2.tolist()) # camera_matrix
print(D2.tolist()) # dist_coefs