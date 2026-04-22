import numpy as np


def to_grayscale(image):
    # Si ya está en gris, regresar igual
    if len(image.shape) == 2:
        return image

    # Fórmula estándar
    gray = 0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]

    return gray.astype(np.uint8)