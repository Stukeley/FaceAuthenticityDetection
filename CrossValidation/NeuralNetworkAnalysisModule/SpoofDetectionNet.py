# Projekt rozpoznawania autentyczności twarzy
# Niniejszy plik zawiera architekturę sieci neuronowej do określania autentyczności twarzy
# Autor: Rafał Klinowski
import torch.nn as nn
import torch.nn.functional as F


class SpoofDetectionNet(nn.Module):
    def __init__(self):
        super(SpoofDetectionNet, self).__init__()
        # CONV => RELU => CONV => RELU => POOL
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)

        # CONV => RELU => CONV => RELU => POOL
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.25)

        # FC => RELU
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.5)

        # Warstwa wyjściowa
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        # CONV => RELU => CONV => RELU => POOL
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # CONV => RELU => CONV => RELU => POOL
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # FC => RELU
        x = self.flatten(x)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout3(x)

        # Klasyfikator Softmax
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
