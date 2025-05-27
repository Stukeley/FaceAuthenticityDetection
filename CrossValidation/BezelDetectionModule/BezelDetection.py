import cv2
import numpy as np

# Module 1 - Bezel Detection
# Only the algorithm, no training
def detect_bezel(image, face) -> float:
    # Image processing
    img_downscaled = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(img_downscaled, cv2.COLOR_BGR2GRAY)

    # New face position
    (x, y, w, h) = face
    scale_factor_X = 256 / image.shape[1]
    scale_factor_Y = 256 / image.shape[0]

    x = int(x * scale_factor_X)
    y = int(y * scale_factor_Y)
    w = int(w * scale_factor_X)
    h = int(h * scale_factor_Y)

    # 1. Declare initial parameters
    # 2. Check for bezel in "up"
    # 3. Gradual change of parameters to minimum threshold
    # 4. Repeat step for the remaining three edges
    # [PARAMETERS]
    bezels_found = 0
    max_bezel_size = 30
    min_bezel_size = 8
    gray_threshold = 28

    x_start = x + w//4
    y_start = 0
    x_end = x + 3*w//4
    y_end = y
    current_bezel_size = max_bezel_size
    candidates = []

    while current_bezel_size >= min_bezel_size:
        # Choose a "strip" with a width of current_bezel_size
        bezel = img_gray[y_start:y_start + current_bezel_size, x_start:x_end]
        # Check if the average pixel value in the strip is less than the threshold
        average = np.mean(bezel)
        if average <= gray_threshold:
            candidates.append((x_start, y_start, x_end, y_start + current_bezel_size, average))

        # Move the strip
        y_start += 1

        if y_start + current_bezel_size >= y_end:
            current_bezel_size -= 1
            y_start = 0

    # Check if a bezel was found
    if len(candidates) > 0:
        bezels_found += 1

    # Search in "down"
    x_start = x + w // 4
    y_start = y + h
    x_end = x + 3 * w // 4
    y_end = 256
    current_bezel_size = max_bezel_size
    candidates = []

    while current_bezel_size >= min_bezel_size:
        bezel = img_gray[y_start:y_start + current_bezel_size, x_start:x_end]
        average = np.mean(bezel)
        if average <= gray_threshold:
            candidates.append((x_start, y_start, x_end, y_start + current_bezel_size, average))

        y_start += 1

        if y_start + current_bezel_size >= y_end:
            current_bezel_size -= 1
            y_start = y

    if len(candidates) > 0:
        bezels_found += 1

    # Search in "left"
    x_start = 0
    y_start = y + h // 4
    x_end = x
    y_end = y + 3 * h // 4
    current_bezel_size = max_bezel_size
    candidates = []

    while current_bezel_size >= min_bezel_size:
        bezel = img_gray[y_start:y_end, x_start:x_start + current_bezel_size]
        average = np.mean(bezel)
        if average <= gray_threshold:
            candidates.append((x_start, y_start, x_start + current_bezel_size, y_end, average))

        x_start += 1

        if x_start + current_bezel_size >= x_end:
            current_bezel_size -= 1
            x_start = 0

    if len(candidates) > 0:
        bezels_found += 1

    # Search in "right"
    x_start = x + w
    y_start = y + h // 4
    x_end = 256
    y_end = y + 3 * h // 4
    current_bezel_size = max_bezel_size
    candidates = []

    while current_bezel_size >= min_bezel_size:
        bezel = img_gray[y_start:y_end, x_start:x_start + current_bezel_size]
        average = np.mean(bezel)
        if average <= gray_threshold:
            candidates.append((x_start, y_start, x_start + current_bezel_size, y_end, average))

        x_start += 1

        if x_start + current_bezel_size >= x_end:
            current_bezel_size -= 1
            x_start = x

    if len(candidates) > 0:
        bezels_found += 1

    # Check if at least 2 bezels were found
    # Probability is based on the amount of bezels found
    probability = bezels_found / 4.0

    return probability
