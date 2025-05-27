import cv2
import numpy as np

# Four metrics
# Normalize them first
# https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html

def correlation(hist1, hist2):
    return 1 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)    # [0; 1]

def chi_square(hist1, hist2):
    chisquare = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    chisquare = 1 / (1 + np.exp(-chisquare))
    return chisquare    # [0; 1]


def intersection(hist1, hist2):
    intersect = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
    intersect = 1 / (1 + np.exp(-intersect))
    return 1 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT) # [0; 1]

def bhattacharyya(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA) # [0; 1]

def calculate_distance(hist1, hist2):
    hist1 = hist1.astype('float32')
    hist2 = hist2.astype('float32')
    return correlation(hist1, hist2) + chi_square(hist1, hist2) + intersection(hist1, hist2) + bhattacharyya(hist1, hist2)