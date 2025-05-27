# Projekt rozpoznawania autentyczności twarzy
# Niniejszy plik zawiera architekturę sieci neuronowej do określenia oceny prawdopodobieństwa podstawienia twarzy
# Na podstawie uzyskanych częściowych prawdopodobieństw
# Autor: Rafał Klinowski
import torch
import torch.nn as nn

# Parameter to control the amount of inputs to the NN
N_INPUTS = 4


class ProbabilityNN(nn.Module):
    def __init__(self):
        super(ProbabilityNN, self).__init__()
        self.fc1 = nn.Linear(N_INPUTS, 10)
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
