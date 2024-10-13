# Projekt rozpoznawania autentyczności twarzy
# W tym pliku znajduje się funkcja obliczająca współrzędne i rozmiar otoczenia twarzy na zdjęciu.
# Autor: Rafał Klinowski


# Funkcja obliczająca współrzędne i rozmiar otoczenia twarzy na zdjęciu.
def get_face_context(img, face) -> dict:
    # 1. Obliczenie proporcji twarzy w stosunku do całego obrazu
    # 2. Obliczenie współrzędnych punktów w bezpośrednim otoczeniu twarzy
    # 3. Obliczenie współrzędnych punktów w dalszym otoczeniu twarzy
    # 4. Zwrócenie współrzędnych z funkcji
    x, y, w, h = face
    img_h, img_w, _ = img.shape

    # Obliczenie proporcji twarzy w stosunku do całego obrazu
    face_proportion = (w * h) / (img_h * img_w)

    # Uznajemy, że bezpośrednie otoczenie twarzy to 40% wysokości i szerokości twarzy
    # [PARAMETR]
    context_proportion = 0.40

    # Obliczenie współrzędnych punktów w bezpośrednim otoczeniu twarzy
    rectangle_point = (x - (context_proportion * w), y - (context_proportion * h))
    rectangle_width = w + 2 * (context_proportion * w)
    rectangle_height = h + 2 * (context_proportion * h)

    # Obliczenie współrzędnych punktów w dalszym otoczeniu twarzy
    rectangle_point_far = (x - (2 * context_proportion * w), y - (2 * context_proportion * h))
    rectangle_width_far = w + 4 * (context_proportion * w)
    rectangle_height_far = h + 4 * (context_proportion * h)

    # Sprawdzenie, czy punkty nie wychodzą poza obraz
    if rectangle_point[0] < 0:
        rectangle_point = (0, rectangle_point[1])
    if rectangle_point[1] < 0:
        rectangle_point = (rectangle_point[0], 0)
    if rectangle_point[0] + rectangle_width > img_w:
        rectangle_width = img_w - rectangle_point[0]
    if rectangle_point[1] + rectangle_height > img_h:
        rectangle_height = img_h - rectangle_point[1]

    if rectangle_point_far[0] < 0:
        rectangle_point_far = (0, rectangle_point_far[1])
    if rectangle_point_far[1] < 0:
        rectangle_point_far = (rectangle_point_far[0], 0)
    if rectangle_point_far[0] + rectangle_width_far > img_w:
        rectangle_width_far = img_w - rectangle_point_far[0]
    if rectangle_point_far[1] + rectangle_height_far > img_h:
        rectangle_height_far = img_h - rectangle_point_far[1]

    # Zwrócenie wartości
    return {
        'face_proportion': face_proportion,
        'rectangle_point': rectangle_point,
        'rectangle_width': rectangle_width,
        'rectangle_height': rectangle_height,
        'rectangle_point_far': rectangle_point_far,
        'rectangle_width_far': rectangle_width_far,
        'rectangle_height_far': rectangle_height_far
    }
