import torch
import torch.nn as nn
from .ProbabilityNN import ProbabilityNN


# Module 6 - final neural network
# Training
def train_probability_nn(df):
    # df - ["Bezel", "Smartphone", "Context", "NN", "Expected"]
    # Data preparation
    inputs = df[["Bezel", "Smartphone", "Context", "NN"]]
    outputs = df["Expected"]

    model = ProbabilityNN()

    dataset = torch.utils.data.TensorDataset(torch.tensor(inputs.values, dtype=torch.float32), torch.tensor(outputs.values, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100

    # Training loop
    for epoch in range(num_epochs):
        for x, y in dataloader:
            y = y.view(-1, 1)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

    return model


# Testing
def probability_nn(results, model):

    # tuple -> tensor
    result_tensor = torch.tensor(results, dtype=torch.float32)

    with torch.no_grad():
        output = model(result_tensor)
        return float(output.item())    # Return a single value - probability of spoofing
