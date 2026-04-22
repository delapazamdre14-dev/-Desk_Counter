import numpy as np

def connected_components(binary):
    height, width = binary.shape
    labels = np.zeros((height, width), dtype=int)

    current_label = 1

    def flood_fill(x, y, label):
        stack = [(x, y)]

        while stack:
            cx, cy = stack.pop()

            if cx < 0 or cx >= height or cy < 0 or cy >= width:
                continue

            if binary[cx, cy] == 0:
                continue

            if labels[cx, cy] != 0:
                continue

            labels[cx, cy] = label

            # vecinos (4-conectividad)
            stack.append((cx+1, cy))
            stack.append((cx-1, cy))
            stack.append((cx, cy+1))
            stack.append((cx, cy-1))

    for i in range(height):
        for j in range(width):
            if binary[i, j] == 255 and labels[i, j] == 0:
                flood_fill(i, j, current_label)
                current_label += 1

    return labels