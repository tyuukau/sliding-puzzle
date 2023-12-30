import pandas as pd
import numpy as np
import warnings


def make_exp_dataset():
    warnings.filterwarnings("ignore", category=FutureWarning)

    df = pd.read_csv("data/input/fifteen-puzzle-6M.csv")
    estimations_df = pd.read_csv("data/eda/estimations.csv")

    df = df.drop_duplicates().reset_index(drop=True)

    merged_df = pd.concat([df, estimations_df], axis=1)
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    selected_costs = [10, 20, 30, 40, 50, 60]
    num_residual_values = 6
    num_nearest_rows = 20

    result_df = pd.DataFrame()

    for cost in selected_costs:
        print("Selecting at", cost)
        # Filter rows by 'cost' and reset the index to avoid reindexing issues
        cost_filtered = merged_df[merged_df["cost"] == cost].copy()

        # Calculate range for residuals based on cost
        min_residual = cost * (-0.5)
        max_residual = 0  # As mentioned, maximum residual is likely 0

        # Create 6 evenly spaced residual values within the range
        residual_values = np.linspace(min_residual, max_residual, num_residual_values)

        # Get 20 nearest rows for each residual value or nearest values
        for residual in residual_values:
            nearest_rows = cost_filtered.iloc[
                (cost_filtered["manhattan_residual"] - residual)
                .abs()
                .argsort()[:num_nearest_rows]
            ]
            result_df = pd.concat([result_df, nearest_rows])

    result_df = result_df[
        [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "cost",
            "manhattan_residual",
        ]
    ]
    result_df = result_df.astype(int)

    result_df.to_csv("data/input/experimental_data.csv", index=False)
