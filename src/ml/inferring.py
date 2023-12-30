import torch
import torch.nn as nn
from torch import device
import torch.nn.functional as F

import numpy as np

from typing import List
from tqdm import tqdm


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


def get_model(model_path: str = "data/models/puzzle_model.pth") -> nn.Module:
    input_size = 256
    hidden_sizes = [1024, 1024, 512, 128, 64]
    model = PuzzleHeuristicModel(input_size, hidden_sizes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

    if "module." in list(checkpoint.keys())[0]:
        new_state_dict = {}
        for key, value in checkpoint.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        checkpoint = new_state_dict

    model.load_state_dict(checkpoint)
    model.eval()

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    return model.to(device)


def infer(input_data: List[int], model: PuzzleHeuristicModel) -> int:
    input_data = torch.tensor(
        np.eye(16)[np.array(input_data).astype(np.int64)].ravel(), dtype=torch.float32
    ).unsqueeze(
        0
    )  # Convert input list to tensor

    with torch.no_grad():
        output = model(input_data)
        predicted_value = output.item()

    return int(round(predicted_value))


def batch_infer(dataframe, model, batch_size=64) -> List[int]:
    input_data = torch.tensor(
        np.eye(16)[dataframe.iloc[:, :16].values.astype(np.int64)].reshape(-1, 16, 256),
        dtype=torch.float32,
    )  # Prepare input data as a tensor

    predicted_values = []

    with torch.no_grad(), tqdm(total=len(input_data)) as pbar:
        for batch in range(0, len(input_data), batch_size):  # assuming batch_size is defined
            batch_input = input_data[batch : batch + batch_size]
            output = model(batch_input)
            predicted_values.extend(output.squeeze().tolist())
            pbar.update(len(batch_input))

    predicted_values = [int(round(item)) for sublist in predicted_values for item in sublist]
    return predicted_values
