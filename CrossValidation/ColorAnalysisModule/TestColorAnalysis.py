import os
import cv2
import cv2.data
import numpy as np

import ColorAnalysis
from ColorAnalysis import train_color_analysis, test_color_analysis

def main():
    # Load data
    auth_data_path = "../../Dane/wszystko/authentic"
    spoof_data_path = "../../Dane/wszystko/spoof"

    # Load authentic images
    authentic_images = []

    files = [f for f in os.listdir(auth_data_path) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png") or f.endswith(".webp")] # Tylko pliki JPG i PNG i WEBP

    for file in files:
        image = cv2.imread(os.path.join(auth_data_path, file))
        authentic_images.append(image)

    # Load spoof images
    spoof_images = []

    files = [f for f in os.listdir(spoof_data_path) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png") or f.endswith(".webp")] # Tylko pliki JPG i PNG i WEBP

    for file in files:
        image = cv2.imread(os.path.join(spoof_data_path, file))
        spoof_images.append(image)

    print("Ilość zdjęć autentycznych: ", len(authentic_images))
    print("Ilość zdjęć podstawionych: ", len(spoof_images))

    # Create X and y from all imgaes
    X = authentic_images + spoof_images
    y = np.concatenate((np.zeros(len(authentic_images)), np.ones(len(spoof_images))))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Split into 50/50
    X_train = X[:int(len(X) * 0.5)]
    y_train = y[:int(len(y) * 0.5)]

    X_test = X[int(len(X) * 0.5):]
    y_test = y[int(len(y) * 0.5):]

    histograms, results = train_color_analysis(X_train, y_train, face_cascade)
    accuracy = 0

    for i in range(len(X_test)):
        img = X_test[i]
        faces = face_cascade.detectMultiScale(img, 1.3, 5, minSize=(30, 30))

        if len(faces) == 0:
            continue

        face = faces[0]

        result = test_color_analysis(img, face, histograms, results)

        if result >= 0.5 and y_test[i] == 1\
                or result < 0.5 and y_test[i] == 0:
            accuracy += 1

    print("Accuracy: ", accuracy / len(X_test))


if __name__ == '__main__':
    main()