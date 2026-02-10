import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import ChessEvalNet

# --------------------
# Config
# --------------------
DATA_PATH = "data/positions.csv"
BATCH_SIZE = 64
EPOCHS = 40
LR = 5e-4
MODEL_PATH = "backend/chess_eval_nn.pt"

# --------------------
# Dataset
# --------------------
class ChessDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        X = df.iloc[:, :-1].values.astype(np.float32)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-6
        self.X = (X - self.mean) / self.std

        y = df.iloc[:, -1].values.astype(np.float32)
        y = np.clip(y, -600, 600)      # clamp extreme tactics
        self.y = y / 100.0             # convert to pawns


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# --------------------
# Training
# --------------------
dataset = ChessDataset(DATA_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ChessEvalNet(input_dim=dataset.X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.MSELoss()

print("Training started...")
for epoch in range(EPOCHS):
    total_loss = 0.0
    for X, y in loader:
        optimizer.zero_grad()
        preds = model(X).squeeze()
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | MSE Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
