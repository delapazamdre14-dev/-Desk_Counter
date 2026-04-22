# Desk Counter

Librería en Python para segmentación y conteo de objetos sobre superficies de trabajo.

##  Características

- Preprocesamiento de imágenes
- Thresholding (Otsu)
- Segmentación por regiones (Connected Components)
- Detección de bordes (Sobel)
- Detección de esquinas (Harris)
- Pipeline configurable

---
# Uso básico
- import cv2
- from desk_counter import count_objects

### Cargar imagen
- img = cv2.imread("imagen.jpg")

### Ejecutar conteo
- result = count_objects(img)
- print("Objetos detectados:", result.count)

---
# Modos de operación
### Threshold (por defecto)
Segmentación basada en umbralización automática.
- result = count_objects(img, mode="threshold")

###  Edge-based

Segmentación basada en detección de bordes (Sobel + operaciones morfológicas).
- result = count_objects(img, mode="edges")

### Uso de filtros personalizados
from desk_counter.preprocessing.filters import gaussian_filter

- filters = [
    {"func": gaussian_filter, "params": {"size": 5, "sigma": 2}}
]
- result = count_objects(img, filters=filters)

### Detección de esquinas (Harris)
- result = count_objects(img, use_corners=True)
- corners = result.corners
---
## 📦 Instalación

```bash
pip install git+https://github.com/TU_USUARIO/Desk_Counter.git
