# Projekt rozpoznawania autentyczności twarzy
# W tym pliku znajduje się funkcja analizująca dystrybucję krawędzi na obrazie w formie histogramu.
# Autor: Rafał Klinowski
import numpy as np


# Funkcja tworząca histogram na podstawie wykrytych krawędzi na obrazie.
def analyze_edge_distribution(sobel):
    edge_data = sobel.flatten()

    # Utworzenie histogramu na podstawie intensywności krawędzi, podzielonego na 16 przedziałów
    # [PARAMETR]
    hist, bins = np.histogram(edge_data, bins=16)

    return hist, bins
