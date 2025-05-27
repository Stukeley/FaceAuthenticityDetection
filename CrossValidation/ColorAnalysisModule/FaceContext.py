# Projekt rozpoznawania autentyczności twarzy
# W tym pliku znajduje się funkcja obliczająca współrzędne i rozmiar otoczenia twarzy na zdjęciu.
# Autor: Rafał Klinowski


# Function calculating the coordinates and size of the face surroundings on the image.
def get_face_context(img, face) -> dict:
    # 1. Calculate the proportions of the face in relation to the entire image
    # 2. Calculate the coordinates of the points in the immediate vicinity of the face
    # 3. Calculate the coordinates of the points in the further vicinity of the face
    # 4. Return the coordinates from the function
    x, y, w, h = face
    img_h, img_w, _ = img.shape

    # Calculating the proportions of the face in relation to the entire image
    face_proportion = (w * h) / (img_h * img_w)

    # Settling that the immediate surroundings of the face are 40% of the height and width of the face
    # [PARAMETER]
    context_proportion = 0.40

    # Calculate the coordinates of the points in the immediate vicinity of the face
    rectangle_point = (x - (context_proportion * w), y - (context_proportion * h))
    rectangle_width = w + 2 * (context_proportion * w)
    rectangle_height = h + 2 * (context_proportion * h)

    # Calculate the coordinates of the points in the further vicinity of the face
    rectangle_point_far = (x - (2 * context_proportion * w), y - (2 * context_proportion * h))
    rectangle_width_far = w + 4 * (context_proportion * w)
    rectangle_height_far = h + 4 * (context_proportion * h)

    # Check if the points do not go beyond the image
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

    # Return values
    return {
        'face_proportion': face_proportion,
        'rectangle_point': rectangle_point,
        'rectangle_width': rectangle_width,
        'rectangle_height': rectangle_height,
        'rectangle_point_far': rectangle_point_far,
        'rectangle_width_far': rectangle_width_far,
        'rectangle_height_far': rectangle_height_far
    }


def get_close_surroundings(img, face):
    face_context = get_face_context(img, face)
    rectangle_point = face_context['rectangle_point']
    rectangle_width = face_context['rectangle_width']
    rectangle_height = face_context['rectangle_height']

    return img[
           int(rectangle_point[1]):int(rectangle_point[1] + rectangle_height),
           int(rectangle_point[0]):int(rectangle_point[0] + rectangle_width)
           ]


def get_face_area(img, face):
    return img[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]


def get_all_image_area(img, face):
    return img
