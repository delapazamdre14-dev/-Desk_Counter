import numpy as np


def dilate(image, kernel_size=3):
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='constant')
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            if np.max(region) > 0:
                output[i, j] = 255

    return output


def erode(image, kernel_size=3):
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='constant')
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            if np.min(region) == 255:
                output[i, j] = 255

    return output


def closing(image, kernel_size=3):
    return erode(dilate(image, kernel_size), kernel_size)