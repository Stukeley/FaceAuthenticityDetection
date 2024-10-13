# Projekt rozpoznawania autentyczności twarzy
# Niniejszy plik zawiera implementację wszystkich niezbędnych elementów oraz warstw
# w systemie rozpoznawania autentyczności twarzy
# Autor: Rafał Klinowski
import cv2
import cv2.data
import time
import numpy as np
import PySimpleGUI as sg
from concurrent.futures import ThreadPoolExecutor
from sys import platform
import ui
import layers
import torch
from ProbabilityNN import ProbabilityNN

# Zmienna globalna przechowująca obiekt kamery
video_capture: cv2.VideoCapture


# Funkcja zwracająca listę dostępnych kamer w systemie.
# Różne podejścia w zależności od systemu operacyjnego.
def get_system_cameras():
    # Windows
    if platform == "win32":
        from pygrabber.dshow_graph import FilterGraph
        devices = FilterGraph().get_input_devices()
        cameras = []

        for index, name in enumerate(devices):
            cameras.append((name, index))

        return cameras
    # MacOS i Linux
    elif platform == "darwin" or platform == "linux" or platform == "linux2":
        # Sprawdzanie dostępnych kamer po indeksach
        cameras = []
        index = 0
        try:
            while index <= 1:
                cap = cv2.VideoCapture(index)
                if cap.read()[0]:
                    cameras.append((f"Kamera {index}", index))
                    index += 1
        finally:
            return cameras


# Funkcja ustawiająca obiekt kamery na podstawie indeksu kamery wybranej przez użytkownika z listy.
def set_video_capture(index=0):
    global video_capture
    video_capture = cv2.VideoCapture(index)
    if not video_capture.isOpened():
        print("Błąd otwarcia kamery - próba otwarcia domyślnej kamery systemowej")
        video_capture = cv2.VideoCapture(0)


# Funkcja zwalniająca obiekt kamery.
def release_video_capture():
    global video_capture
    video_capture.release()


# Funkcja pobierająca pojedynczą klatkę z kamery.
def get_video_frame():
    global video_capture
    ret, frame = video_capture.read()

    if not ret:
        print("Błąd odczytu klatki")
        return None

    return frame


# Funkcja pobierająca pojedynczą klatkę z "lampą błyskową" (rozbłyskiem ekranu) z kamery.
def get_video_frame_with_flash():
    # Wyświetlenie białego, jasnego ekranu na chwilę, następnie zrobienie zdjęcia
    # Rozdzielczość ekranu
    screen_width = 1920
    screen_height = 1080

    # Utworzenie białego okna o rozdzielczości ekranu
    flash = np.ones((screen_height, screen_width, 3), np.uint8)
    flash = flash * 255

    try:
        # Wyświetlenie okna
        cv2.namedWindow("Flash", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Flash", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Flash", flash)
        time.sleep(0.5)

        # Zdjęcie z kamery
        ret, frame = video_capture.read()

        cv2.destroyWindow("Flash")
        cv2.waitKey(1)

        if not ret:
            print("Błąd odczytu klatki")
            return None

        return frame
    except Exception as e:
        print(f"Błąd: {e}")
        return None


# Funkcja wykrywająca twarz na zdjęciu wejściowym (klatce z kamery).
def detect_face(frame):
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # [PARAMETRY]
    faces = face_classifier.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return faces


# Funkcja przyjmująca obraz z kamery oraz wykrytą twarz,
# rozdzielająca wywołania poszczególnych warstw systemu w sposób równoległy.
def schedule_authenticity_check(image, face, image_flash, face_flash, index) -> (str, bool, float):
    # Rozdzielenie zadań w zależności od przekazanego indeksu
    if index == 0:
        print("Wykrycie ramki")
        return layers.detect_bezel(image, face)

    elif index == 1:
        print("Wykrycie smartfona")
        return layers.detect_smartphone(image, face)

    elif index == 2:
        print("Analiza kontekstu")
        return layers.analyze_context(image, face)

    elif index == 3:
        print("Analiza oświetlenia w serii")
        # W przypadku, gdy nie uda się uzyskać zdjęcia z "lampą błyskową"
        # Zwracamy informację o niepowodzeniu (nie oznacza ani twarzy autentycznej, ani podstawionej)
        if image_flash is None or face_flash is None:
            return "-LIGHTING_INFO-", None, -1.0

        return layers.analyze_lighting(image, face, image_flash, face_flash)

    elif index == 4:
        print("Analiza twarzy przy pomocy sieci neuronowej")
        return layers.analyze_face(image, face)


# Funkcja obliczająca prawdopodobieństwo, że twarz na wejściu jest podstawiona,
# na podstawie uzyskanych z warstw wyników oraz ich wag.
# Wersja 2: wykorzystanie prostej sieci neuronowej.
def calculate_spoof_probability(results) -> float:

    # Załadowanie sieci neuronowej
    model = ProbabilityNN()
    model.load_state_dict(torch.load("data/probability_nn.pth"))
    model.eval()

    # Przygotowanie danych wejściowych
    results = [probability for (_, _, probability) in results]
    results = torch.tensor(results, dtype=torch.float32).view(1, -1)

    # Uzyskanie wyniku z sieci neuronowej
    with torch.no_grad():
        output = model(results)
        return output.item()    # Zwracanie pojedynczej liczby - prawdobodobieństwa


def main():
    # Inicjalizacja UI
    # Lista urządzeń wideo dostępnych w systemie
    cameras = get_system_cameras()
    window = ui.generate_user_interface(cameras)

    # Inicjalizacja warstw systemu
    layers.initialize_layers()

    # Wstępne ustawienie kamery na pierwszą pozycję z listy
    _, first_camera_index = cameras[0]
    set_video_capture(first_camera_index)

    # Odczekanie 2 sekundy na kamerę
    time.sleep(2)

    # Pętla UI
    while True:
        event, values = window.read(timeout=100)

        if event == sg.WIN_CLOSED:
            break

        # Wybór kamery przez użytkownika
        elif event == "-CAMERA_SELECT-":
            chosen_camera = values["-CAMERA_SELECT-"]
            chosen_camera_index = 0
            for camera in cameras:
                if camera[0] == chosen_camera:
                    chosen_camera_index = camera[1]
                    break
            set_video_capture(chosen_camera_index)
            # Odczekanie 2 sekundy na kamerę
            time.sleep(2)

            # Aktualizacja informacji o kamerze
            ui.update_single_info(window, "-CAMERA_INFO-", chosen_camera)

        elif event == "-CHECK_AUTHENTICITY-":
            print("Sprawdzanie autentyczności")
            # DEBUG: Pomiar czasu
            time_start = time.time()
            # Wyczyszczenie poprzednich informacji
            ui.clear_info(window)

            frame = get_video_frame()
            faces = detect_face(frame)

            if faces is None or len(faces) == 0:
                # Aktualizacja UI - powiadomienie o braku wykrytej twarzy
                ui.update_single_info(window, "-FACE_INFO-", "Brak")
                continue

            # W przypadku wykrycia kilku twarzy, wybieramy największą (dodatkowe zabezpieczenie przed szumem)
            face = max(faces, key=lambda f: f[2] * f[3])

            # Pobranie klatki po rozbłysku ekranu
            # Operacja ta musi być wykonana z wątku głównego
            frame_flash = get_video_frame_with_flash()
            face_flash = None

            if frame_flash is None:
                # Aktualizacja UI - powiadomienie o błędzie
                ui.update_single_info(window, "-LIGHTING_INFO-", "Niepowodzenie")
            else:
                faces_flash = detect_face(frame_flash)

                if faces_flash is None or len(faces_flash) == 0:
                    # Aktualizacja UI - powiadomienie o błędzie
                    ui.update_single_info(window, "-LIGHTING_INFO-", "Niepowodzenie")
                    face_flash = None
                else:
                    face_flash = max(faces_flash, key=lambda f: f[2] * f[3])

            # Skopiowanie i przekazanie danych do wątków
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(
                    schedule_authenticity_check, frame.copy(), face, frame_flash, face_flash, i) for i in range(5)]
                results = [future.result() for future in futures]

            # Rozpakowanie uzyskanych wyników
            results_dict = {"-CAMERA_AVAILABLE-": "Dostępna", "-FACE_INFO-": "Wykryta"}
            for i, result in enumerate(results):
                layer_id, layer_result, layer_probability = result
                # Obsłużenie przypadku, gdy warstwa zwróciła None
                if layer_result is None:
                    results_dict[layer_id] = "Niepowodzenie"
                elif layer_result is True:
                    results_dict[layer_id] = "Podstawiona"
                else:
                    results_dict[layer_id] = "Autentyczna"

            # Obliczenie prawdopodobieństwa, że twarz jest podstawiona
            probability = calculate_spoof_probability(results)

            # Końcowa ocena na podstawie prawdopodobieństwa
            if probability > 0.5:
                results_dict["-FACE_SCORE-"] = "Podstawiona"
                results_dict["-PROBABILITY-"] = f"{probability:.2f}"    # Prawdopodobieństwo, że twarz jest podstawiona
            else:
                results_dict["-FACE_SCORE-"] = "Autentyczna"
                results_dict["-PROBABILITY-"] = f"{probability:.2f}"    # Prawdopodobieństwo, że twarz jest autentyczna

            # Aktualizacja informacji na ekranie
            ui.update_photo_info(window, results_dict)
            time2 = time.time()
            print(f"Czas: {time2 - time_start}")

        frame = get_video_frame()

        if frame is not None:
            # Skalowanie obrazu do wyświetlenia w interfejsie
            frame_ui = cv2.resize(frame, (480, 270))

            faces = detect_face(frame_ui)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame_ui, (x, y), (x+w, y+h), (0, 0, 255), 2)

            # Aktualizacja UI - wyświetlenie obrazu z kamery
            image = cv2.imencode('.png', frame_ui)[1].tobytes()
            ui.update_camera_feed(window, image)

            # Odczekanie przed kolejną iteracją
            cv2.waitKey(1)
        else:
            # Powiadomienie o błędzie
            print("Błąd odczytu klatki")
            ui.update_single_info(window, "-CAMERA_AVAILABLE-", "Brak")

    window.close()
    release_video_capture()


if __name__ == '__main__':
    main()
