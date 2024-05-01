import numpy as np

def data(n):
    v = np.ones((256, 256))
    for i in range(256):
        for j in range(256):
            if (128 - i) ** 2 + (128 - j) ** 2 < 76 ** 2:
                v[i][j] = ((128 - i) ** 2 + (128 - j) ** 2) / n ** 2 + 1
            else:
                v[i][j] = n ** 2 / ((128 - i) ** 2 + (128 - j) ** 2) + 1
    return v

