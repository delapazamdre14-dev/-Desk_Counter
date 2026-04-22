import numpy as np


# =========================
# THRESHOLD BASE (REUTILIZABLE)
# =========================
def threshold_binary(image, thresh, invert=False):
    binary = np.zeros_like(image)

    if invert:
        binary[image < thresh] = 255
    else:
        binary[image > thresh] = 255

    return binary


# =========================
# THRESHOLD FIJO
# =========================
def threshold(image, thresh=127, invert=False):
    return threshold_binary(image, thresh, invert)


# =========================
# OTSU (CÁLCULO DEL UMBRAL)
# =========================
def compute_otsu_threshold(image):
    img = image.astype(np.uint8)

    hist = np.zeros(256)
    for value in img.flatten():
        hist[value] += 1

    total_pixels = img.size

    sum_total = 0
    for i in range(256):
        sum_total += i * hist[i]

    sum_bg = 0
    weight_bg = 0
    max_variance = 0
    best_thresh = 0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue

        weight_fg = total_pixels - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * hist[t]

        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if variance > max_variance:
            max_variance = variance
            best_thresh = t

    return best_thresh


# =========================
# OTSU COMPLETO
# =========================
def threshold_otsu(image, invert=False):
    thresh = compute_otsu_threshold(image)
    return threshold_binary(image, thresh, invert)