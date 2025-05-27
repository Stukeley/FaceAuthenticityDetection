import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
from .SpoofDetectionNet import SpoofDetectionNet


# Module 4
# Training
def train_analyze_face(X_train, y_train, face_cascade):
    # Preprocessing
    # Data augmentation to simulate real-life conditions
    aug = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),  # Random mirror flip
        transforms.RandomRotation(5),  # Random rotation by angle from [-5, 5]
        transforms.ColorJitter(brightness=0.1, saturation=0.1),  # Change brightness and saturation
    ])

    resize_trf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
    ])

    tensor_trf = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Function to cut out the area close to the face
    def get_image_part(img_p):
        faces = face_cascade.detectMultiScale(img_p, 1.3, 5)
        if len(faces) == 0:
            return None
        (x, y, w, h) = faces[0]
        x = max(0, x - w // 3)
        y = max(0, y - h // 3)
        w = min(img_p.shape[1], w * 5 // 3)
        h = min(img_p.shape[0], h * 5 // 3)
        return img_p[y:y+h, x:x+w]

    # Preparing sets
    X_train_aug = []
    y_train_aug = []

    for i in range(len(X_train)):
        image = X_train[i]
        is_face_spoofed = y_train[i] # 0 - authentic, 1 - spoof

        for j in range(3):
            augmented_image = aug(image)
            augmented_image_np = np.array(augmented_image)
            face_part = get_image_part(augmented_image_np)

            if face_part is None:
                continue

            resized_face = resize_trf(face_part)

            # Normalizacja
            resized_face = np.array(resized_face) / 255.0

            # Na tensor
            resized_face = tensor_trf(resized_face)

            X_train_aug.append(resized_face)
            y_train_aug.append(is_face_spoofed)

    X_train_tensor = torch.stack(X_train_aug)
    X_train_tensor = torch.tensor(X_train_tensor, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_aug, dtype=torch.long)

    train_loader = torch.utils.data.DataLoader(list(zip(X_train_tensor, y_train_tensor)), batch_size=32, shuffle=True)

    # Model
    model = SpoofDetectionNet()
    criterion = nn.CrossEntropyLoss()
    INIT_LR = 1e-3
    EPOCHS = 10
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR, weight_decay=INIT_LR / EPOCHS)
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Return the trained model
    return model


# Algorithm
def analyze_face(image, face, model):
    # Preparation of the input image
    # Isolation of the area around the face
    (x, y, w, h) = face
    x = max(0, x - w // 3)
    y = max(0, y - h // 3)
    w = min(image.shape[1], w * 5 // 3)
    h = min(image.shape[0], h * 5 // 3)
    face_area = image[y:y + h, x:x + w]

    trf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    face_area_processed = trf(face_area).unsqueeze(0)

    model.eval()    # !!!!! VERY IMPORTANT TO SET THE MODEL TO EVALUATION MODE !!!!!

    # Passing the image to the neural network
    with torch.no_grad():
        output = model(face_area_processed)
        # 'result' will be equal to 0 or 1, where 1 means a spoofed face
        result = torch.argmax(output, dim=1).item()

        # Calculation of the probability that the face is authentic [0] or spoofed [1]
        probabilities = torch.nn.functional.softmax(output, dim=1)

        # Return probability of spoofed face
        return float(probabilities[0][1].item())
