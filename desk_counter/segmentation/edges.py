import numpy as np
from desk_counter.preprocessing.filters import apply_convolution


def sobel_edges(image):
    Kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    Ky = np.array([
        [-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1]
    ], dtype=np.float32)

    gx = apply_convolution(image, Kx, keep_float=True)
    gy = apply_convolution(image, Ky, keep_float=True)

    magnitude = np.sqrt(gx**2 + gy**2)

    magnitude = magnitude / np.max(magnitude) if np.max(magnitude) != 0 else magnitude
    magnitude = (magnitude * 255).astype(np.uint8)

    return magnitude