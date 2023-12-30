import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np

import os
import logging
from tqdm import tqdm

from ..utils import make_dir


class PuzzleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state = torch.tensor(self.data.iloc[idx, :-1].values, dtype=torch.float32)
        state_one_hot = torch.tensor(
            np.eye(16)[state.to(torch.int64)].ravel(), dtype=torch.float32
        )
        cost = torch.tensor(self.data.iloc[idx, -1], dtype=torch.float32)
        return state_one_hot, cost


class PuzzleHeuristicModel(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(PuzzleHeuristicModel, self).__init__()
        all_sizes = [input_size] + hidden_sizes + [1]
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(all_sizes[i], all_sizes[i + 1]) for i in range(len(all_sizes) - 1)]
        )

    def forward(self, x):
        for layer in self.hidden_layers[:-1]:
            x = F.relu(layer(x))
        x = self.hidden_layers[-1](x)
        return x


def train(
    input_file_path: str, save_model_dir: str, run_name: str, n: int, should_stratify: bool
):
    working_dir = save_model_dir + run_name
    make_dir(working_dir)

    print("Starting training...")

    log_file = f"{working_dir}/logs.txt"
    if os.path.exists(log_file):
        os.remove(log_file)
    logging.basicConfig(
        handlers=[
            logging.FileHandler(log_file),  # Output to file
            logging.StreamHandler(),  # Output to console
        ],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(f"Logs and weights will be found in {working_dir}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Loading dataset...")
    # Load the data
    file_path = input_file_path

    if n > 0:
        data = pd.read_csv(file_path, nrows=n)
    else:
        data = pd.read_csv(file_path)

    data = data[data["cost"] < 70]
    logging.info("Done.")

    logging.info("Splitting the data into train, set, val sets...")

    if should_stratify:  # assuming 'should_stratify' is a boolean variable
        train_data, remaining_data = train_test_split(
            data, test_size=0.3, stratify=data["cost"], random_state=42
        )
        test_data, val_data = train_test_split(
            remaining_data, test_size=0.5, stratify=remaining_data["cost"], random_state=42
        )
    else:
        train_data, remaining_data = train_test_split(data, test_size=0.3, random_state=42)
        test_data, val_data = train_test_split(remaining_data, test_size=0.5, random_state=42)

    logging.info("Done.")

    # Display the sizes of each split
    logging.info(f"Train data size: {len(train_data)}")
    logging.info(f"Test data size: {len(test_data)}")
    logging.info(f"Validation data size: {len(val_data)}")

    batch_size = 64
    train_dataset = PuzzleDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = PuzzleDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = PuzzleDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_size = 256
    hidden_sizes = [1024, 1024, 512, 128, 64]
    model = PuzzleHeuristicModel(input_size, hidden_sizes)

    logging.info(f"Hidden sizes: {hidden_sizes}")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_epochs = 10
    best_loss = float("inf")

    best_model_folder = working_dir + "/immediate/"
    if not os.path.exists(best_model_folder):
        os.makedirs(best_model_folder)

    best_model_path = best_model_folder + "best_puzzle_model.pth"

    logging.info("Training...")
    LOG_INTERVAL = 10

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", file=open(os.devnull, "w")
        )
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()
            if progress_bar.n % LOG_INTERVAL == 0:
                logging.info(str(progress_bar))

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            progress_bar = tqdm(val_loader, desc=f"Validation")
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets.view(-1, 1)).item()
                if progress_bar.n % LOG_INTERVAL == 0:
                    logging.info(str(progress_bar))
            val_loss /= len(val_loader)  # Average validation loss

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}")

        # Save the best model based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    logging.info("Done.")
    logging.info("Testing...")

    if torch.cuda.device_count() > 1:
        best_model = nn.DataParallel(model)
    else:
        best_model = model
    best_model.to(device)
    best_model.load_state_dict(torch.load(best_model_path))

    # Testing
    test_loss = 0.0
    best_model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = best_model(inputs)
            test_loss += criterion(outputs, targets.view(-1, 1)).item()
    test_loss /= len(test_loader)  # Average test loss

    logging.info(f"Final Test Loss: {test_loss}")
    logging.info("Done.")
