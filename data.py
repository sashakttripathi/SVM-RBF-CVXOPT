import numpy as np


# A gaussian sample
def get_sample():
    x1 = np.array([[8, 6], [8, 8], [8, 10], [9, 5], [11, 5],
                   [12, 6], [11, 8], [10, 10]])
    y1 = np.ones(x1.shape[0])

    x2 = np.array([[9, 2], [13, 4], [14, 8], [12, 11], [10, 13],
                   [6, 9], [6, 5]])
    y2 = -1 * np.ones(x2.shape[0])

    return np.vstack((x1, x2)), np.hstack((y1, y2))
