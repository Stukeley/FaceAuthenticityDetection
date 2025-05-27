import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from .BezelDetectionNet import BezelDetectionNet


# Module 2 - Smartphone Detection
# Training
def train_detect_smartphone(X_train, y_train):
    model = BezelDetectionNet()

    # Data preprocessing
    trf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    X_train = [trf(image) for image in X_train]
    y_train = torch.tensor(y_train, dtype=torch.long)

    train_loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=32, shuffle=True)

    # Model training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.train()

    # Training loop
    for epoch in range(10):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Return the trained model
    return model


# Algorithm
def detect_smartphone(image, face, model):
    # Image processing
    trf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    image_processed = trf(image).unsqueeze(0)

    # Model evaluation
    with torch.no_grad():
        output = model(image_processed)
        # 'result' will be 0 for original and 1 for spoof
        result = torch.argmax(output, dim=1).item()

        # Calculate probabilities of the face being authentic [0] or spoof [1]
        probabilities = torch.nn.functional.softmax(output, dim=1)

        # Return probability of the face being spoofed
        return float(probabilities[0][1].item())
