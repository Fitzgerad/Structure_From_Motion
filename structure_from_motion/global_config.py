import os
import numpy as np

# DIR_PATH points to the file where we save all the photos.
DIR_PATH = os.path.normpath('../images/2')

# INTRINSIC_MATRIX is the intrinsic matrix of the cameras by which we took all the photos.
# INTRINSIC_LIST = [[ 424.9,     0, 176.0],
#                   [     0, 424.9, 144.0],
#                   [     0,     0,   1.0]]
# statue
INTRINSIC_LIST = [[ 3410.99,       0, 3129.34],
                  [       0, 3412.37, 2059.72],
                  [       0,       0,     1.0]]
INTRINSIC_MATRIX = np.array(INTRINSIC_LIST)

PROJECTION_LIST = [[     1,     0,     0,     0],
                   [     0,     1,     0,     0],
                   [     0,     0,     1,     0]]
PROJECTION_MATRIX = np.array(PROJECTION_LIST)

W_LIST = [[     0,    -1,     0],
          [     1,     0,     0],
          [     0,     0,     1]]
W_MATRIX = np.array(W_LIST)

# RANSAC_SIZE is the amount of feature points used to compute the Fundamental Matrix in
# one Ransac iteration.
FUNDAMENTAL_RANSAC_SIZE = 20

FUNDAMENTAL_RANSAC_THRESHOLD = 0.01
