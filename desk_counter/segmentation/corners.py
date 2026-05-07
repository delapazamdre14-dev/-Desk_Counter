import numpy as np
from desk_counter.preprocessing.filters import apply_convolution


def harris_corners(image, k=0.04, threshold=0.01):
    img = image.astype(np.float32)

    
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

    Ix = apply_convolution(img, Kx, keep_float=True)
    Iy = apply_convolution(img, Ky, keep_float=True)

   
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    
    kernel = np.ones((3, 3)) / 9.0

    Sxx = apply_convolution(Ixx, kernel, keep_float=True)
    Syy = apply_convolution(Iyy, kernel, keep_float=True)
    Sxy = apply_convolution(Ixy, kernel, keep_float=True)

    
    det = (Sxx * Syy) - (Sxy ** 2)
    trace = Sxx + Syy

    R = det - k * (trace ** 2)

    
    R = R / np.max(R) if np.max(R) != 0 else R

    
    corners = np.zeros_like(R)
    corners[R > threshold] = 255

    return corners.astype(np.uint8), R
