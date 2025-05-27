# Projekt rozpoznawania autentyczności twarzy
# W tym pliku znajduje się funkcja analizująca dystrybucję krawędzi na obrazie w formie histogramu.
# Autor: Rafał Klinowski
import numpy as np


# Function creating a histogram based on detected edges on the image.
def analyze_edge_distribution(sobel):
    edge_data = sobel.flatten()

    # Create a histogram based on the intensity of the edges, divided into 64 intervals
    # [PARAMETER]
    hist, bins = np.histogram(edge_data, bins=64)   # TODO: test different number of bins

    return hist, bins
