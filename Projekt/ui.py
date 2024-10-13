# Projekt rozpoznawania autentyczności twarzy
# Niniejszy plik zawiera deklarację interfejsu użytkownika aplikacji
# Autor: Rafał Klinowski
import PySimpleGUI as sg


# Funkcja generująca interfejs użytkownika w podstawowym stanie.
# Zgodnie z zaprojektowanym prototypem.
def generate_user_interface(cameras: list[tuple[str, int]]):
    # Utworzenie listy nazw kamer
    cameras = [camera[0] for camera in cameras]

    # Stałe interfejsu - kolory i czcionki
    COLOR_BACKGROUND = "#F5F5F5"
    COLOR_BUTTON = "#CCCCCC"
    COLOR_TEXT = "#000000"
    FONT_DEFAULT = ("Arial", 12)
    FONT_DEFAULT_BOLD = ("Arial", 12, "bold")

    # Utworzenie układu okna
    layout = [[
        sg.Column([
            [sg.Text("Kamera", background_color=COLOR_BACKGROUND, text_color=COLOR_TEXT, key="-CAMERA_INFO-",
                     justification="center", size=(30, 1), font=FONT_DEFAULT_BOLD, pad=(10, 0))],
            [sg.Image(filename="", size=(480, 270), background_color=COLOR_BACKGROUND, key="-CAMERA-")],
        ], vertical_alignment='top', background_color=COLOR_BACKGROUND),
        sg.Column([
            [sg.Combo(cameras, default_value=cameras[0], background_color=COLOR_BUTTON, key="-CAMERA_SELECT-", size=(30, 1),
                      enable_events=True, font=FONT_DEFAULT, readonly=True)],

            [sg.Button("Sprawdź", button_color=COLOR_BUTTON, key="-CHECK_AUTHENTICITY-", size=(30, 1),
                       enable_events=True, font=FONT_DEFAULT)],

            [sg.Text("Kamera:", size=(15, 1), background_color=COLOR_BACKGROUND, text_color=COLOR_TEXT, font=FONT_DEFAULT),
                sg.Text('', size=(20, 1), background_color=COLOR_BACKGROUND, text_color=COLOR_TEXT, key="-CAMERA_AVAILABLE-",
                        font=FONT_DEFAULT)],

            [sg.Text("Twarz:", background_color=COLOR_BACKGROUND, size=(15, 1), text_color=COLOR_TEXT, font=FONT_DEFAULT),
                sg.Text('', size=(20, 1), background_color=COLOR_BACKGROUND, text_color=COLOR_TEXT, key="-FACE_INFO-",
                        font=FONT_DEFAULT)],

            [sg.Text("Smartfon:", size=(15, 1), background_color=COLOR_BACKGROUND, text_color=COLOR_TEXT, font=FONT_DEFAULT),
                sg.Text('', size=(20, 1), background_color=COLOR_BACKGROUND, text_color=COLOR_TEXT, key="-SMARTPHONE_INFO-",
                        font=FONT_DEFAULT)],

            [sg.Text("Ramka:", size=(15, 1), background_color=COLOR_BACKGROUND, text_color=COLOR_TEXT, font=FONT_DEFAULT),
                sg.Text('', size=(20, 1), background_color=COLOR_BACKGROUND, text_color=COLOR_TEXT, key="-BEZEL_INFO-",
                        font=FONT_DEFAULT)],

            [sg.Text("Analiza kontekstu:", size=(15, 1), background_color=COLOR_BACKGROUND, text_color=COLOR_TEXT,
                     font=FONT_DEFAULT),
                sg.Text('', size=(20, 1), background_color=COLOR_BACKGROUND, text_color=COLOR_TEXT, key="-CONTEXT_INFO-",
                        font=FONT_DEFAULT)],

            [sg.Text("Analiza oświetlenia:", size=(15, 1), background_color=COLOR_BACKGROUND, text_color=COLOR_TEXT,
                     font=FONT_DEFAULT),
                sg.Text('', size=(20, 1), background_color=COLOR_BACKGROUND, text_color=COLOR_TEXT, key="-LIGHTING_INFO-",
                        font=FONT_DEFAULT)],

            [sg.Text("Sieć neuronowa:", size=(15, 1), background_color=COLOR_BACKGROUND, text_color=COLOR_TEXT,
                     font=FONT_DEFAULT),
                sg.Text('', size=(20, 1), background_color=COLOR_BACKGROUND, text_color=COLOR_TEXT, key="-NN_INFO-",
                        font=FONT_DEFAULT)],

            [sg.HorizontalSeparator()],

            [sg.Text("Ocena twarzy:", size=(15, 1), background_color=COLOR_BACKGROUND, text_color=COLOR_TEXT,
                     font=FONT_DEFAULT_BOLD),
                sg.Text('', size=(20, 1), background_color=COLOR_BACKGROUND, text_color=COLOR_TEXT, key="-FACE_SCORE-",
                        font=FONT_DEFAULT_BOLD)],

            [sg.Text("Prawd. podstawienia:", size=(20, 1), background_color=COLOR_BACKGROUND, text_color=COLOR_TEXT,
                        font=FONT_DEFAULT),
                sg.Text('', size=(15, 1), background_color=COLOR_BACKGROUND, text_color=COLOR_TEXT, key="-PROBABILITY-",
                            font=FONT_DEFAULT)],

        ], element_justification="right", vertical_alignment='top', background_color=COLOR_BACKGROUND)
        ]]

    # Utworzenie okna ze zdefiniowanym układem
    window = sg.Window("Rozpoznawanie Autentyczości Twarzy", layout, size=(800, 350), background_color=COLOR_BACKGROUND)
    return window


# Funkcja usuwająca z ekranu wszystkie informacje (np. z poprzednich wywołań programu).
def clear_info(window):
    window["-CAMERA_INFO-"].update('', text_color="black")
    window["-FACE_INFO-"].update('', text_color="black")
    window["-SMARTPHONE_INFO-"].update('', text_color="black")
    window["-BEZEL_INFO-"].update('', text_color="black")
    window["-CONTEXT_INFO-"].update('', text_color="black")
    window["-LIGHTING_INFO-"].update('', text_color="black")
    window["-NN_INFO-"].update('', text_color="black")
    window["-FACE_SCORE-"].update('', text_color="black")
    window["-PROBABILITY-"].update('', text_color="black")


# Funkcja aktualizująca wyświetlany obraz z kamery.
def update_camera_feed(window, image):
    # Aktualizacja wyświetlanego obrazu
    window["-CAMERA-"].update(data=image)


# Funkcja aktualizująca pojedynczą informację na ekranie.
def update_single_info(window, key: str, value: str):
    # Zmiana koloru w zależności od wartości
    if key == "-CAMERA_INFO-":
        window[key].update(value)
    elif key == "-FACE_INFO-":
        if value == "Wykryta":
            window[key].update(value, text_color="green")
        else:
            window[key].update(value, text_color="red")
    elif key == "-CAMERA_AVAILABLE-":
        if value == "Dostępna":
            window[key].update(value, text_color="green")
        else:
            window[key].update(value, text_color="red")


# Funkcja aktualizująca informacje o kamerze.
def update_photo_info(window, photo_info: dict[str, str]):
    # Aktualizacja poszczególnych informacji wyświetlanych na ekranie
    # window["-CAMERA_INFO-"].update(photo_info["-CAMERA_INFO-"])
    # window["-FACE_INFO-"].update(photo_info["-FACE_INFO-"])
    # window["-SMARTPHONE_INFO-"].update(photo_info["-SMARTPHONE_INFO-"])
    # window["-BEZEL_INFO-"].update(photo_info["-BEZEL_INFO-"])
    # window["-CONTEXT_INFO-"].update(photo_info["-CONTEXT_INFO-"])
    # window["-LIGHTING_INFO-"].update(photo_info["-LIGHTING_INFO-"])
    # window["-NN_INFO-"].update(photo_info["-NN_INFO-"])
    # window["-FACE_SCORE-"].update(photo_info["-FACE_SCORE-"])
    # window["-PROBABILITY-"].update(photo_info["-PROBABILITY-"])

    # Aktualizacja kolorów w zależności od wyniku
    CAMERA_AVAILABLE = "Dostępna"
    FACE_DETECTED = "Wykryta"
    FACE_AUTHENTIC = "Autentyczna"
    FACE_SPOOF = "Podstawiona"
    LAYER_ERROR = "Niepowodzenie"

    if photo_info["-CAMERA_AVAILABLE-"] == CAMERA_AVAILABLE:
        window["-CAMERA_AVAILABLE-"].update(photo_info["-CAMERA_AVAILABLE-"], text_color="green")
    else:
        window["-CAMERA_AVAILABLE-"].update(photo_info["-CAMERA_AVAILABLE-"], text_color="red")

    if photo_info["-FACE_INFO-"] == FACE_DETECTED:
        window["-FACE_INFO-"].update(photo_info["-FACE_INFO-"], text_color="green")
    else:
        window["-FACE_INFO-"].update(photo_info["-FACE_INFO-"], text_color="red")

    if photo_info["-SMARTPHONE_INFO-"] == FACE_SPOOF:
        window["-SMARTPHONE_INFO-"].update("Wykryty", text_color="red")
    else:
        window["-SMARTPHONE_INFO-"].update("Brak", text_color="green")

    if photo_info["-BEZEL_INFO-"] == FACE_SPOOF:
        window["-BEZEL_INFO-"].update("Wykryta", text_color="red")
    else:
        window["-BEZEL_INFO-"].update("Brak", text_color="green")

    if photo_info["-CONTEXT_INFO-"] == FACE_SPOOF:
        window["-CONTEXT_INFO-"].update("Podstawiona", text_color="red")
    else:
        window["-CONTEXT_INFO-"].update(FACE_AUTHENTIC, text_color="green")

    if photo_info["-LIGHTING_INFO-"] == LAYER_ERROR:
        window["-LIGHTING_INFO-"].update("Niepowodzenie", text_color="black")
    elif photo_info["-LIGHTING_INFO-"] == FACE_SPOOF:
        window["-LIGHTING_INFO-"].update("Podstawiona", text_color="red")
    else:
        window["-LIGHTING_INFO-"].update(FACE_AUTHENTIC, text_color="green")

    if photo_info["-NN_INFO-"] == FACE_SPOOF:
        window["-NN_INFO-"].update("Podstawiona", text_color="red")
    else:
        window["-NN_INFO-"].update(FACE_AUTHENTIC, text_color="green")

    if photo_info["-FACE_SCORE-"] == FACE_SPOOF:
        window["-FACE_SCORE-"].update("Podstawiona", text_color="red")
    else:
        window["-FACE_SCORE-"].update(FACE_AUTHENTIC, text_color="green")

    window["-PROBABILITY-"].update(photo_info["-PROBABILITY-"])

