import pandas as pd

from ..utils import create_scatterplot_and_save, make_dir
from ..ml import get_model, batch_infer


def conduct_overestimation_experiment(
    input_file_path: str, experiment_folder_path: str, n: int = 0
) -> None:
    make_dir(experiment_folder_path)

    if n > 0:
        df = pd.read_csv(input_file_path, nrows=n)
    else:
        df = pd.read_csv(input_file_path)

    df = df.drop_duplicates()

    model = get_model()

    predicted_costs = batch_infer(df, model)
    df["predicted_cost"] = predicted_costs

    columns_to_use = [str(i) for i in range(16)]  # Create a list of column names '0' to '15'
    df.drop_duplicates(subset=columns_to_use, inplace=True)

    len(df)

    df["ann_residual"] = df["predicted_cost"] - df["cost"]

    print("Creating residual plot for ANN distance")
    create_scatterplot_and_save(
        df=df,
        x="cost",
        y="ann_residual",
        file_path=experiment_folder_path + "ann_residual_plot.png",
    )

    over_df = df[df["ann_residual"] > 0]
    print("Overestimated instance percentage: ", len(over_df) / len(df))

    df.to_csv(experiment_folder_path + "experiment1_results.csv")
