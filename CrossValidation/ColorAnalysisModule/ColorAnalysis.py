# Module to compare histograms of area close to the face between each other
# R, G and B separately
# kNN based on distance between histograms

import cv2

import CalculateDistance
import FaceContext
from CalculateDistance import calculate_distance
from FaceContext import get_face_context

def train_color_analysis(X_train, y_train, face_cascade):
    # Preprocess images - get close area to the face
    X_processed = []

    for img in X_train:
        faces = face_cascade.detectMultiScale(img, 1.3, 5, minSize=(30, 30))

        if len(faces) == 0:
            continue

        face = faces[0]

        face_context = get_face_context(img, face)

        rectangle_point = (int(face_context['rectangle_point'][0]), int(face_context['rectangle_point'][1]))
        rectangle_width = int(face_context['rectangle_width'])
        rectangle_height = int(face_context['rectangle_height'])

        context_area_close = img[rectangle_point[1]:rectangle_point[1] + rectangle_height,
                             rectangle_point[0]:rectangle_point[0] + rectangle_width]

        X_processed.append(context_area_close)

    # Calculate histograms for each image separately for each channel
    histograms = []
    results = []

    img_size = (256, 256)
    bins = 32

    for i in range(len(X_processed)):

        img = X_processed[i]

        img = cv2.resize(img, img_size)

        hist_R = cv2.calcHist([img], [0], None, [bins], [0, 256])
        hist_G = cv2.calcHist([img], [1], None, [bins], [0, 256])
        hist_B = cv2.calcHist([img], [2], None, [bins], [0, 256])

        histograms.append([hist_R, hist_G, hist_B])
        results.append(y_train[i])

    return histograms, results


def test_color_analysis(image, face, histograms, results):
    # Create 3 pairs of histograms-results

    hist_R = []
    results_R = []

    hist_G = []
    results_G = []

    hist_B = []
    results_B = []

    for i in range(len(histograms)):
        hist_R.append(histograms[i][0])
        results_R.append(results[i])

        hist_G.append(histograms[i][1])
        results_G.append(results[i])

        hist_B.append(histograms[i][2])
        results_B.append(results[i])

    # Calculate histograms for the image
    img_size = (256, 256)
    bins = 32

    face_context = get_face_context(image, face)

    rectangle_point = (int(face_context['rectangle_point'][0]), int(face_context['rectangle_point'][1]))
    rectangle_width = int(face_context['rectangle_width'])
    rectangle_height = int(face_context['rectangle_height'])

    context_area_close = image[rectangle_point[1]:rectangle_point[1] + rectangle_height,
                            rectangle_point[0]:rectangle_point[0] + rectangle_width]

    context_area_close = cv2.resize(context_area_close, img_size)

    new_hist_R = cv2.calcHist([context_area_close], [0], None, [bins], [0, 256])
    new_hist_G = cv2.calcHist([context_area_close], [1], None, [bins], [0, 256])
    new_hist_B = cv2.calcHist([context_area_close], [2], None, [bins], [0, 256])

    # Create 3 manual kNNs, each for a different channel
    total_auth = 0
    total_spoof = 0

    # R
    distances = []
    for i in range(len(hist_R)):
        hist = hist_R[i]
        result = results_R[i]
        distance = calculate_distance(hist, new_hist_R)
        distances.append((distance, result))

    distances.sort(key=lambda x: x[0])
    k = 3
    distances = distances[:k]

    for distance, result in distances:
        if result == 0:
            total_auth += 1
        else:
            total_spoof += 1

    # G
    distances = []
    for i in range(len(hist_G)):
        hist = hist_G[i]
        result = results_G[i]
        distance = calculate_distance(hist, new_hist_G)
        distances.append((distance, result))

    distances.sort(key=lambda x: x[0])
    k = 3
    distances = distances[:k]

    for distance, result in distances:
        if result == 0:
            total_auth += 1
        else:
            total_spoof += 1

    # B
    distances = []

    for i in range(len(hist_B)):
        hist = hist_B[i]
        result = results_B[i]
        distance = calculate_distance(hist, new_hist_B)
        distances.append((distance, result))

    distances.sort(key=lambda x: x[0])
    k = 3
    distances = distances[:k]

    for distance, result in distances:
        if result == 0:
            total_auth += 1
        else:
            total_spoof += 1

    # Calculate final result
    result = total_spoof / (total_auth + total_spoof)

    return result
