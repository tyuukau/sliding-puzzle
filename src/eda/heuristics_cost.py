import pandas as pd
from tqdm import tqdm

from ..utils import create_scatterplot_and_save, make_dir
from ..heuristics import manhattan_distance, mistile_distance
from ..state import State


normal_state = State(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)


def _calculate_manhattan(row) -> int:
    state_values = row[0:16].tolist()  # Extract values from columns 0 to 15
    state = State(*state_values)
    manhattan_estimation = manhattan_distance(state, normal_state)
    return manhattan_estimation


def _calculate_mistile(row) -> int:
    state_values = row[0:16].tolist()  # Extract values from columns 0 to 15
    state = State(*state_values)
    mistile_estimation = mistile_distance(state, normal_state)
    return mistile_estimation


def evaluate_heuristics_on_dataset(
    input_file_path: str, eda_folder_dir: str, n: int = 0
) -> None:
    make_dir(eda_folder_dir)

    tqdm.pandas()

    file_path = input_file_path

    if n > 0:
        df = pd.read_csv(file_path, nrows=n)
    else:
        df = pd.read_csv(file_path)

    df = df.drop_duplicates()

    df["manhattan_estimation"] = df.progress_apply(_calculate_manhattan, axis=1)
    df["mistile_estimation"] = df.progress_apply(_calculate_mistile, axis=1)

    df.drop(columns=list(str(i) for i in range(16)), inplace=True)

    # Assuming df contains 'cost' and 'manhattan_estimation' columns
    # Calculate residuals
    df["manhattan_residual"] = df["manhattan_estimation"] - df["cost"]
    df["mistile_residual"] = df["mistile_estimation"] - df["cost"]

    output_file = eda_folder_dir + "estimations.csv"
    df.to_csv(output_file)

    # Create a residual plot
    create_scatterplot_and_save(
        df=df,
        x="cost",
        y="manhattan_residual",
        file_path=eda_folder_dir + "manhattan_residual_plot.png",
    )

    create_scatterplot_and_save(
        df=df,
        x="cost",
        y="mistile_residual",
        file_path=eda_folder_dir + "mistile_residual_plot.png",
    )

    print("Done.")
