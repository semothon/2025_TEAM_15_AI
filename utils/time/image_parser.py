import cv2
import numpy as np

def get_schedule_mask(image_path):
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    cell_w = w // 5  # 요일
    cell_h = h // 12  # 시간

    mask = np.zeros((12, 5), dtype=np.uint8)

    for row in range(12):
        for col in range(5):
            x1, y1 = col * cell_w, row * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h
            cell = img[y1:y2, x1:x2]
            b, g, r = cell.mean(axis=(0, 1))

            if not ((r > 200 and g > 200 and b > 200) or (r < 50 and g < 50 and b < 50)):
                mask[row, col] = 1

    return mask