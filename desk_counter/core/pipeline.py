from desk_counter.preprocessing.grayscale import to_grayscale
from desk_counter.segmentation.thresholding import threshold_otsu
from desk_counter.segmentation.connected_components import connected_components
from desk_counter.segmentation.edges import sobel_edges
from desk_counter.segmentation.morphology import closing
from desk_counter.segmentation.corners import harris_corners  # 🔥 NUEVO

import numpy as np


class DeskCounterResult:
    def __init__(self, count, labels, corners=None):
        self.count = count
        self.labels = labels
        self.corners = corners  # 🔥 NUEVO

    def show(self):
        print(f"Objetos detectados: {self.count}")


def count_objects(image, filters=None, invert=False, mode="threshold", use_corners=False):
    # 1. Preprocesamiento
    gray = to_grayscale(image)
    processed = gray

    # Filtros
    if filters is not None:
        for f in filters:
            if callable(f):
                processed = f(processed)
            elif isinstance(f, dict):
                func = f["func"]
                params = f.get("params", {})
                processed = func(processed, **params)

    # =========================
    # MODO THRESHOLD
    # =========================
    if mode == "threshold":
        binary = threshold_otsu(processed, invert=invert)

    # =========================
    # MODO EDGES
    # =========================
    elif mode == "edges":
        edges = sobel_edges(processed)
        edges_bin = (edges > 100).astype(np.uint8) * 255
        binary = closing(edges_bin, kernel_size=5)

    else:
        raise ValueError("Modo no válido. Usa 'threshold' o 'edges'.")

    # 3. Etiquetado
    labels = connected_components(binary)

    # 4. Conteo
    count = np.max(labels)

    # =========================
    # 🔥 CORNERS OPCIONAL
    # =========================
    corners = None
    if use_corners:
        corners, _ = harris_corners(processed)

    return DeskCounterResult(count, labels, corners)