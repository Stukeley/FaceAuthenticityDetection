# Projekt rozpoznawania autentyczności twarzy
# W tym pliku znajdują się funkcje odpowiadające poszczególnym warstwom systemu.
# Każda z warstw zwraca parę wartości - identyfikator warstwy
# oraz informację o tym, czy warstwa zwróciła pozytywny wynik.
# Autor: Rafał Klinowski
import numpy as np
import cv2
import torch
from torchvision import transforms
from BezelDetectionNet import BezelDetectionNet
from SpoofDetectionNet import SpoofDetectionNet
from FaceContext import get_face_context
from EdgeDistribution import analyze_edge_distribution


# Zmienne ładowane jeden raz podczas inicjalizacji programu
bezelDetectionNet = BezelDetectionNet()
spoofDetectionNet = SpoofDetectionNet()
data_X = None
data_Y = None


# Funkcja inicjalizująca warstwy systemu.
def initialize_layers():
    global data_X, data_Y
    data_X = np.load("data/histogram_data_far_X.npz")
    data_Y = np.load("data/histogram_data_far_Y.npz")

    global bezelDetectionNet, spoofDetectionNet
    bezelDetectionNet.load_state_dict(torch.load("data/bezelai_model.pth"))
    spoofDetectionNet.load_state_dict(torch.load("data/model_closeup.pth"))
    bezelDetectionNet.eval()
    spoofDetectionNet.eval()


# Funkcja wykrywająca obecność ramek urządzeń elektronicznych wokół twarzy na zdjęciu wejściowym.
def detect_bezel(image, face) -> (str, bool, float):
    # Przetwarzanie wstępne obrazu wejściowego
    img_downscaled = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(img_downscaled, cv2.COLOR_BGR2GRAY)

    # Uzyskanie nowej pozycji twarzy po przeskalowaniu
    (x, y, w, h) = face
    scale_factor_X = 256 / image.shape[1]
    scale_factor_Y = 256 / image.shape[0]

    x = int(x * scale_factor_X)
    y = int(y * scale_factor_Y)
    w = int(w * scale_factor_X)
    h = int(h * scale_factor_Y)

    # 1. Ustalenie parametrów początkowych
    # 2. Sprawdzenie obecności ramki w "górę"
    # 3. Stopniowa zmiana parametrów do minimalnego progu
    # 4. Powtórzenie kroku dla pozostałych trzech krawędzi
    # [PARAMETRY]
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
        # Wybieramy "pasek" o szerokości current_bezel_size
        bezel = img_gray[y_start:y_start + current_bezel_size, x_start:x_end]
        # Sprawdzamy, czy średnia wartość pikseli w pasku jest mniejsza niż próg
        average = np.mean(bezel)
        if average <= gray_threshold:
            candidates.append((x_start, y_start, x_end, y_start + current_bezel_size, average))

        # Przesunięcie paska
        y_start += 1

        if y_start + current_bezel_size >= y_end:
            current_bezel_size -= 1
            y_start = 0

    # Sprawdzenie, czy znaleziono ramkę
    if len(candidates) > 0:
        bezels_found += 1

    # Wyszukiwanie w "dół"
    x_start = x + w // 4
    y_start = y + h
    x_end = x + 3 * w // 4
    y_end = 256
    current_bezel_size = max_bezel_size
    candidates = []

    while current_bezel_size >= min_bezel_size:
        # Wybieramy "pasek" o szerokości current_bezel_size
        bezel = img_gray[y_start:y_start + current_bezel_size, x_start:x_end]
        # Sprawdzamy, czy średnia wartość pikseli w pasku jest mniejsza niż próg
        average = np.mean(bezel)
        if average <= gray_threshold:
            candidates.append((x_start, y_start, x_end, y_start + current_bezel_size, average))

        # Przesunięcie paska
        y_start += 1

        if y_start + current_bezel_size >= y_end:
            current_bezel_size -= 1
            y_start = y

    # Sprawdzenie, czy znaleziono ramkę
    if len(candidates) > 0:
        bezels_found += 1

    # Wyszukiwanie w "lewo"
    x_start = 0
    y_start = y + h // 4
    x_end = x
    y_end = y + 3 * h // 4
    current_bezel_size = max_bezel_size
    candidates = []

    while current_bezel_size >= min_bezel_size:
        # Wybieramy "pasek" o szerokości current_bezel_size
        bezel = img_gray[y_start:y_end, x_start:x_start + current_bezel_size]
        # Sprawdzamy, czy średnia wartość pikseli w pasku jest mniejsza niż próg
        average = np.mean(bezel)
        if average <= gray_threshold:
            candidates.append((x_start, y_start, x_start + current_bezel_size, y_end, average))

        # Przesunięcie paska
        x_start += 1

        if x_start + current_bezel_size >= x_end:
            current_bezel_size -= 1
            x_start = 0

    # Sprawdzenie, czy znaleziono ramkę
    if len(candidates) > 0:
        bezels_found += 1

    # Wyszukiwanie w "prawo"
    x_start = x + w
    y_start = y + h // 4
    x_end = 256
    y_end = y + 3 * h // 4
    current_bezel_size = max_bezel_size
    candidates = []

    while current_bezel_size >= min_bezel_size:
        # Wybieramy "pasek" o szerokości current_bezel_size
        bezel = img_gray[y_start:y_end, x_start:x_start + current_bezel_size]
        # Sprawdzamy, czy średnia wartość pikseli w pasku jest mniejsza niż próg
        average = np.mean(bezel)
        if average <= gray_threshold:
            candidates.append((x_start, y_start, x_start + current_bezel_size, y_end, average))

        # Przesunięcie paska
        x_start += 1

        if x_start + current_bezel_size >= x_end:
            current_bezel_size -= 1
            x_start = x

    # Sprawdzenie, czy znaleziono ramkę
    if len(candidates) > 0:
        bezels_found += 1

    # Sprawdzenie, czy znaleziono co najmniej 2 ramki
    if bezels_found >= 2:
        return "-BEZEL_INFO-", True, 1.0
    else:
        return "-BEZEL_INFO-", False, 0.0


# Funkcja wykrywająca obecność smartfona na zdjęciu wejściowym przy pomocy sztucznej inteligencji.
def detect_smartphone(image, face) -> (str, bool, float):
    global bezelDetectionNet

    # Przetwarzanie wstępne obrazu wejściowego
    trf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    image_processed = trf(image).unsqueeze(0)

    # Przekazanie obrazu do sieci neuronowej
    with torch.no_grad():
        output = bezelDetectionNet(image_processed)
        # 'result' będzie równy 0 lub 1, gdzie 1 oznacza obecność smartfona
        result = torch.argmax(output, dim=1).item()

        # Obliczenie prawdopodobieństwa, że twarz jest podstawiona
        probabilities = torch.nn.functional.softmax(output, dim=1)

        # Zwrócenie wyniku
        return "-SMARTPHONE_INFO-", result == 1, probabilities[0][1].item()


# Funkcja analizująca otoczenie twarzy na zdjęciu wejściowym w porównaniu do oczekiwanej dystrybucji.
def analyze_context(image, face) -> (str, bool, float):
    global data_X, data_Y

    # Obliczenie otoczenia twarzy
    context = get_face_context(image, face)

    # Wycięcie 'dalszego' obszaru twarzy
    (x, y, w, h) = face
    rectangle_point_far = (int(context['rectangle_point_far'][0]), int(context['rectangle_point_far'][1]))
    rectangle_width_far = int(context['rectangle_width_far'])
    rectangle_height_far = int(context['rectangle_height_far'])

    context_area_far = image[rectangle_point_far[1]:rectangle_point_far[1] + rectangle_height_far,
                       rectangle_point_far[0]:rectangle_point_far[0] + rectangle_width_far]

    # Przetwarzanie wstępne obrazu wejściowego
    context_area_far = cv2.resize(context_area_far, (256, 256))
    context_area_far = cv2.cvtColor(context_area_far, cv2.COLOR_BGR2GRAY)

    # Sobel X
    sobel_X = cv2.Sobel(context_area_far, cv2.CV_64F, 1, 0, ksize=3)
    sobel_X = cv2.convertScaleAbs(sobel_X)

    # Sobel Y
    sobel_Y = cv2.Sobel(context_area_far, cv2.CV_64F, 0, 1, ksize=3)
    sobel_Y = cv2.convertScaleAbs(sobel_Y)

    # Analiza krawędzi - utworzenie histogramów
    hist_X, bins_X = analyze_edge_distribution(sobel_X)
    hist_Y, bins_Y = analyze_edge_distribution(sobel_Y)

    # Porównanie z oczekiwaną dystrybucją
    # [PARAMETRY]
    max_deviation = 1.5   # Ilość odchyleń standardowych (w dowolną stronę), które są uznawane za odbiegające od normy
    num_deviations = 0  # Ilość wykrytych, odbiegających od normy wartości
    min_num_deviations = 3  # Minimalna ilość wykrytych, odbiegających od normy wartości, aby uznać obraz za niespójny

    # Dla X
    z_score_X = (hist_X - data_X['hist_mean']) / data_X['hist_std']
    num_deviations += np.sum(np.abs(z_score_X) > max_deviation)

    # Dla Y
    z_score_Y = (hist_Y - data_Y['hist_mean']) / data_Y['hist_std']
    num_deviations += np.sum(np.abs(z_score_Y) > max_deviation)

    # Zwrócenie wyniku
    probability = 1.0 if num_deviations >= min_num_deviations else 0.0
    return "-CONTEXT_INFO-", bool(num_deviations >= min_num_deviations), probability


# Funkcja analizująca oświetlenie twarzy na zdjęciu wejściowym.
def analyze_lighting(image, face, flash_image, flash_image_face) -> (str, bool, float):
    # Podejście: wycięcie fragmentu zdjęcia (nie całego), przeskalowanie, przekonwertowanie na CIELUV i porównanie
    # [PARAMETRY]
    img_size = (256, 256)

    # Wycięcie fragmentu zdjęcia
    (x, y, w, h) = face
    face_area = image[y:y + h, x:x + w]

    (x, y, w, h) = flash_image_face
    flash_face_area = flash_image[y:y + h, x:x + w]

    # Przetworzenie obrazu
    face_area = cv2.resize(face_area, img_size)
    flash_face_area = cv2.resize(flash_face_area, img_size)

    face_area = cv2.cvtColor(face_area, cv2.COLOR_BGR2Luv)
    flash_face_area = cv2.cvtColor(flash_face_area, cv2.COLOR_BGR2Luv)

    # Obliczenie średniej wartości kanału L
    face_area_L = np.mean(face_area[:, :, 0])
    flash_face_area_L = np.mean(flash_face_area[:, :, 0])

    diff = cv2.absdiff(face_area_L, flash_face_area_L)

    # Różnica dla obszaru twarzy
    mean_diff_face = diff.mean()

    # Sprawdzenie zdjęcia bez twarzy
    # 1. Resize, 2. Obliczenie współczynnika proporcji, 3. Obliczenie średniej "ręcznie" dla obszaru poza twarzą
    img_size = (512, 512)
    image_no_face = image.copy()
    image_no_face = cv2.resize(image_no_face, img_size)
    image_no_face = cv2.cvtColor(image_no_face, cv2.COLOR_BGR2Luv)
    flash_image_no_face = flash_image.copy()
    flash_image_no_face = cv2.resize(flash_image_no_face, img_size)
    flash_image_no_face = cv2.cvtColor(flash_image_no_face, cv2.COLOR_BGR2Luv)

    proportion_image = (image.shape[0] / img_size[0], image.shape[1] / img_size[1])
    proportion_flash_image = (flash_image.shape[0] / img_size[0], flash_image.shape[1] / img_size[1])

    face_position = (face[0] / proportion_image[0], face[1] / proportion_image[1], face[2] / proportion_image[0], face[3] / proportion_image[1])
    flash_face_position = (flash_image_face[0] / proportion_flash_image[0], flash_image_face[1] / proportion_flash_image[1], flash_image_face[2] / proportion_flash_image[0], flash_image_face[3] / proportion_flash_image[1])

    # Obliczenie obszaru poza twarzą
    mean_noflash = 0
    count_noflash = 0
    mean_flash = 0
    count_flash = 0
    for i in range (0, 512):
        for j in range(0, 512):
            if (i >= face_position[1] and i <= face_position[1] + face_position[3] and j >= face_position[0] and j <= face_position[0] + face_position[2]):
                continue
            else:
                mean_noflash += image_no_face[i, j, 0]
                count_noflash += 1

            if (i >= flash_face_position[1] and i <= flash_face_position[1] + flash_face_position[3] and j >= flash_face_position[0] and j <= flash_face_position[0] + flash_face_position[2]):
                continue
            else:
                mean_flash += flash_image_no_face[i, j, 0]
                count_flash += 1

    mean_noflash /= count_noflash
    mean_flash /= count_flash

    diff = cv2.absdiff(mean_noflash, mean_flash)
    mean_diff_image = diff.mean()

    # Jeżeli różnica dla zdjęcia bez twarzy jest większa, niż dla twarzy
    # To oznacza, że twarz może być podstawiona
    if mean_diff_image > mean_diff_face:
        return "-LIGHTING_INFO-", True, 1.0
    else:
        return "-LIGHTING_INFO-", False, 0.0


# Funkcja analizująca twarz na zdjęciu wejściowym przy pomocy sieci neuronowej.
def analyze_face(image, face) -> (str, bool, float):
    global spoofDetectionNet

    # Przetwarzanie wstępne obrazu wejściowego
    # Wydzielenie obszaru wokół twarzy
    (x, y, w, h) = face
    x = max(0, x - w // 3)
    y = max(0, y - h // 3)
    w = min(image.shape[1], w * 5 // 3)
    h = min(image.shape[0], h * 5 // 3)
    face_area = image[y:y + h, x:x + w]

    # Przetworzenie obrazu
    trf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    face_area_processed = trf(face_area).unsqueeze(0)

    # Przekazanie obrazu do sieci neuronowej
    with torch.no_grad():
        output = spoofDetectionNet(face_area_processed)
        # 'result' będzie równy 0 lub 1, gdzie 1 oznacza twarz podstawioną
        result = torch.argmax(output, dim=1).item()

        # Obliczenie prawdopodobieństwa, że twarz jest podstawiona
        probabilities = torch.nn.functional.softmax(output, dim=1)

        # Zwrócenie wyniku
        return "-NN_INFO-", result == 1, probabilities[0][1].item()
