import numpy as np


def normalize(image):
    return image.astype(np.float32) / 255.0

def denormalize(image):
    image = image / np.max(image) if np.max(image) != 0 else image
    return np.uint8(image * 255)


def log_transform(image, c=1):
    img = normalize(image)
    result = c * np.log(1 + img)
    return denormalize(result)

def gamma_transform(image, gamma=1.0, c=1):
    img = normalize(image)
    result = c * (img ** gamma)
    return denormalize(result)

def invert(image):
    return 255 - image


def gaussian_kernel(size=5, sigma=1):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)

    return kernel

def apply_convolution(image, kernel, keep_float=False):
    img = image.astype(np.float32)

    h, w = img.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)

    if keep_float:
        return output 
    else:
        return np.uint8(np.clip(output, 0, 255))  


def gaussian_filter(image, size=5, sigma=1):
    kernel = gaussian_kernel(size, sigma)
    return apply_convolution(image, kernel)
