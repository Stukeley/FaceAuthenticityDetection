import cv2
import numpy as np
from .CalculateDistance import calculate_distance
from .EdgeDistribution import analyze_edge_distribution
from .FaceContext import get_face_context


# Module 3 - Context Analysis
# Training
def train_analyze_context(X_train, y_train, face_cascade):
    histograms_close_auth = []
    histograms_far_auth = []
    histograms_close_spoof = []
    histograms_far_spoof = []

    for i in range(len(X_train)):
        img = X_train[i]
        print("Image size: ", img.shape)
        is_face_spoofed = bool(y_train[i])

        faces = face_cascade.detectMultiScale(img, 1.3, 5, minSize=(30, 30))

        if len(faces) == 0:
            continue

        face = faces[0]

        face_context = get_face_context(img, face)

        (x, y, w, h) = face

        # Close context of the face
        rectangle_point = (int(face_context['rectangle_point'][0]), int(face_context['rectangle_point'][1]))
        rectangle_width = int(face_context['rectangle_width'])
        rectangle_height = int(face_context['rectangle_height'])

        context_area_close = img[rectangle_point[1]:rectangle_point[1] + rectangle_height,
                             rectangle_point[0]:rectangle_point[0] + rectangle_width]
        context_area_close_gray = cv2.cvtColor(context_area_close, cv2.COLOR_RGB2GRAY)

        # Sobel - close
        context_area_edges_close_X = cv2.Sobel(context_area_close_gray, cv2.CV_64F, 1, 0, ksize=3)
        context_area_edges_close_X = cv2.convertScaleAbs(context_area_edges_close_X)
        context_area_edges_close_Y = cv2.Sobel(context_area_close_gray, cv2.CV_64F, 0, 1, ksize=3)
        context_area_edges_close_Y = cv2.convertScaleAbs(context_area_edges_close_Y)
        context_area_edges_close = np.sqrt(context_area_edges_close_X ** 2 + context_area_edges_close_Y ** 2)

        # Far context of the face
        rectangle_point_far = (int(face_context['rectangle_point_far'][0]), int(face_context['rectangle_point_far'][1]))
        rectangle_width_far = int(face_context['rectangle_width_far'])
        rectangle_height_far = int(face_context['rectangle_height_far'])

        context_area_far = img[rectangle_point_far[1]:rectangle_point_far[1] + rectangle_height_far, rectangle_point_far[0]:rectangle_point_far[0] + rectangle_width_far]
        context_area_far_gray = cv2.cvtColor(context_area_far, cv2.COLOR_RGB2GRAY)

        # Sobel - far
        context_area_edges_far_X = cv2.Sobel(context_area_far_gray, cv2.CV_64F, 1, 0, ksize=3)
        context_area_edges_far_X = cv2.convertScaleAbs(context_area_edges_far_X)
        context_area_edges_far_Y = cv2.Sobel(context_area_far_gray, cv2.CV_64F, 0, 1, ksize=3)
        context_area_edges_far_Y = cv2.convertScaleAbs(context_area_edges_far_Y)
        context_area_edges_far = np.sqrt(context_area_edges_far_X ** 2 + context_area_edges_far_Y ** 2)

        # Analyze edge intensity
        hist_close, bins = analyze_edge_distribution(context_area_edges_close)
        hist_far, bins = analyze_edge_distribution(context_area_edges_far)

        # Normalize histograms
        hist_close = hist_close / np.sum(hist_close)
        hist_far = hist_far / np.sum(hist_far)

        # Append histograms into the right list
        if not is_face_spoofed:
            histograms_close_auth.append(hist_close)
            histograms_far_auth.append(hist_far)
        else:
            histograms_close_spoof.append(hist_close)
            histograms_far_spoof.append(hist_far)

    return histograms_close_auth, histograms_far_auth, histograms_close_spoof, histograms_far_spoof


# Algorithm
def analyze_context(image, face, data_close_auth, data_far_auth, data_close_spoof, data_far_spoof):
    # Calculate face context
    context = get_face_context(image, face)

    # Select 'close' area of the face
    (x, y, w, h) = face
    rectangle_point = (int(context['rectangle_point'][0]), int(context['rectangle_point'][1]))
    rectangle_width = int(context['rectangle_width'])
    rectangle_height = int(context['rectangle_height'])

    context_area_close = image[rectangle_point[1]:rectangle_point[1] + rectangle_height,
                         rectangle_point[0]:rectangle_point[0] + rectangle_width]
    context_area_close_gray = cv2.cvtColor(context_area_close, cv2.COLOR_RGB2GRAY)

    # Sobel - close
    context_area_edges_close_X = cv2.Sobel(context_area_close_gray, cv2.CV_64F, 1, 0, ksize=3)
    context_area_edges_close_X = cv2.convertScaleAbs(context_area_edges_close_X)
    context_area_edges_close_Y = cv2.Sobel(context_area_close_gray, cv2.CV_64F, 0, 1, ksize=3)
    context_area_edges_close_Y = cv2.convertScaleAbs(context_area_edges_close_Y)
    context_area_edges_close = np.sqrt(context_area_edges_close_X ** 2 + context_area_edges_close_Y ** 2)

    # Select 'far' area of the face
    (x, y, w, h) = face
    rectangle_point_far = (int(context['rectangle_point_far'][0]), int(context['rectangle_point_far'][1]))
    rectangle_width_far = int(context['rectangle_width_far'])
    rectangle_height_far = int(context['rectangle_height_far'])

    context_area_far = image[rectangle_point_far[1]:rectangle_point_far[1] + rectangle_height_far,
                       rectangle_point_far[0]:rectangle_point_far[0] + rectangle_width_far]
    context_area_far_gray = cv2.cvtColor(context_area_far, cv2.COLOR_RGB2GRAY)

    # Sobel - far
    context_area_edges_far_X = cv2.Sobel(context_area_far_gray, cv2.CV_64F, 1, 0, ksize=3)
    context_area_edges_far_X = cv2.convertScaleAbs(context_area_edges_far_X)
    context_area_edges_far_Y = cv2.Sobel(context_area_far_gray, cv2.CV_64F, 0, 1, ksize=3)
    context_area_edges_far_Y = cv2.convertScaleAbs(context_area_edges_far_Y)
    context_area_edges_far = np.sqrt(context_area_edges_far_X ** 2 + context_area_edges_far_Y ** 2)

    # Analyze edge intensity
    hist_close, bins = analyze_edge_distribution(context_area_edges_close)
    hist_far, bins = analyze_edge_distribution(context_area_edges_far)

    # Normalize histograms
    hist_close = hist_close / np.sum(hist_close)
    hist_far = hist_far / np.sum(hist_far)

    # kNN "on the fly" for histograms based on 4 metrics: correlation, chi-square, intersection, Bhattacharyya
    # Metrics: https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
    distances = []
    k = 3   # [PARAMETER]

    # Separate analysis for close and far context
    for hist_class, histograms in [("authentic", data_close_auth), ("spoof", data_close_spoof)]:
        for hist in histograms:
            distance = calculate_distance(hist, hist_close)
            distances.append((distance, hist_class))

    distances.sort(key=lambda x: x[0])
    selected_close = distances[:k]

    for hist_class, histograms in [("authentic", data_far_auth), ("spoof", data_far_spoof)]:
        for hist in histograms:
            distance = calculate_distance(hist, hist_far)
            distances.append((distance, hist_class))

    distances.sort(key=lambda x: x[0])
    selected_far = distances[:k]

    # Calculate neighbours
    votes = {"authentic": 0, "spoof": 0}

    for distance, hist_class in selected_close:
        votes[hist_class] += 1

    for distance, hist_class in selected_far:
        votes[hist_class] += 1

    # Calculate probability based on votes
    edges_probability = votes["spoof"] / (votes["authentic"] + votes["spoof"])

    return edges_probability
