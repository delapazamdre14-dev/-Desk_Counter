import cv2
import matplotlib.pyplot as plt

from desk_counter.core.pipeline import count_objects
from desk_counter.preprocessing.filters import gaussian_filter
from desk_counter.preprocessing.grayscale import to_grayscale
from desk_counter.segmentation.thresholding import threshold_otsu
from desk_counter.segmentation.edges import sobel_edges
from desk_counter.segmentation.morphology import closing

# =========================
# Cargar imagen
# =========================
img = cv2.imread("C:/Users/andre/Downloads/Desk.jpg")

if img is None:
    raise ValueError("No se pudo cargar la imagen. Revisa la ruta.")

# =========================
# Configuración
# =========================
filters_config = [
    {"func": gaussian_filter, "params": {"size": 2, "sigma": 4}}
]

invert = True

# =========================
# PIPELINE NORMAL
# =========================
result_thresh = count_objects(
    img,
    filters=filters_config,
    invert=invert,
    mode="threshold"
)

# =========================
# PIPELINE EDGES
# =========================
result_edges = count_objects(
    img,
    filters=filters_config,
    mode="edges"
)

# =========================
# PIPELINE CON CORNERS
# =========================
result_corners = count_objects(
    img,
    filters=filters_config,
    use_corners=True
)

print("Objetos (threshold):", result_thresh.count)
print("Objetos (edges):", result_edges.count)

# =========================
# DEBUG VISUAL
# =========================
gray = to_grayscale(img)

processed = gray
for f in filters_config:
    processed = f["func"](processed, **f["params"])

# ----- threshold pipeline -----
binary_thresh = threshold_otsu(processed, invert=invert)
labels_thresh = result_thresh.labels

# ----- edges pipeline -----
edges = sobel_edges(processed)
edges_bin = (edges > 100).astype("uint8") * 255
closed = closing(edges_bin, kernel_size=5)
labels_edges = result_edges.labels

# ----- corners -----
corners = result_corners.corners

# overlay (corners sobre imagen)
overlay = img.copy()
overlay[corners == 255] = [0, 0, 255]
overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# =========================
# Mostrar resultados
# =========================
plt.figure(figsize=(14, 12))

# ===== FILA 1 =====
plt.subplot(3, 4, 1)
plt.imshow(gray, cmap='gray')
plt.title("Gris")
plt.axis("off")

plt.subplot(3, 4, 2)
plt.imshow(processed, cmap='gray')
plt.title("Filtrada")
plt.axis("off")

plt.subplot(3, 4, 3)
plt.imshow(edges, cmap='gray')
plt.title("Edges (Sobel)")
plt.axis("off")

plt.subplot(3, 4, 4)
plt.imshow(edges_bin, cmap='gray')
plt.title("Edges Binarios")
plt.axis("off")

# ===== FILA 2 =====
plt.subplot(3, 4, 5)
plt.imshow(binary_thresh, cmap='gray')
plt.title("Threshold")
plt.axis("off")

plt.subplot(3, 4, 6)
plt.imshow(labels_thresh, cmap='nipy_spectral')
plt.title(f"Componentes (T): {result_thresh.count}")
plt.axis("off")

plt.subplot(3, 4, 7)
plt.imshow(closed, cmap='gray')
plt.title("Edges Cerrados")
plt.axis("off")

plt.subplot(3, 4, 8)
plt.imshow(labels_edges, cmap='nipy_spectral')
plt.title(f"Componentes (E): {result_edges.count}")
plt.axis("off")

# ===== FILA 3 (CORNERS) =====
plt.subplot(3, 4, 9)
plt.imshow(corners, cmap='gray')
plt.title("Corners (Harris)")
plt.axis("off")

plt.subplot(3, 4, 10)
plt.imshow(overlay_rgb)
plt.title("Corners Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()