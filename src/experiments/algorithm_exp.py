from typing import List
import pandas as pd
from tqdm import tqdm

from multiprocessing import Pool
from functools import partial
import numpy as np

from ..algorithms.abstract_search import InformedSearchAlgorithm
from ..game_config import GameConfig
from ..state import State
from ..algorithms import AStar, GBFS
from ..heuristics import CallableHeuristicClass, manhattan_distance, ann_distance
from ..game import Game, ResultRecord


def play_on_one(
    start_board: List[int], algorithm: InformedSearchAlgorithm, heuristic_func
) -> ResultRecord:
    game_config = GameConfig(
        start_state=State(*start_board),
        goal_state=State(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0),
    )
    algorithm_instance = algorithm(heuristic=heuristic_func)
    g = Game(game_config=game_config, algorithm=algorithm_instance, ignore_solvability=False)

    result_record = g.play()
    return result_record


def conduct_algorithm_heuristic_experiment(
    df: pd.DataFrame,
    algorithm: InformedSearchAlgorithm,
    heuristic: CallableHeuristicClass,
    experiment_folder_path: str,
):
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        start_board = row[:16].tolist()
        result = play_on_one(start_board, algorithm, heuristic)

        # Update DataFrame with the obtained results
        df.at[index, "path_length"] = result.path_length
        df.at[index, "time_cp"] = result.time_cp
        df.at[index, "space_cp"] = result.space_cp
        df.at[index, "time"] = result.time

    df.to_csv(
        experiment_folder_path
        + "algorithm_exp"
        + "_"
        + algorithm.__class__.__name__
        + "_"
        + heuristic.__class__.__name__
        + ".csv"
    )

    return df


def conduct_algorithm_heuristic_experiment_(
    df: pd.DataFrame,
    algorithm: InformedSearchAlgorithm,
    heuristic: CallableHeuristicClass,
):
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        start_board = row[:16].tolist()
        result = play_on_one(start_board, algorithm, heuristic)

        # Update DataFrame with the obtained results
        df.at[index, "path_length"] = result.path_length
        df.at[index, "time_cp"] = result.time_cp
        df.at[index, "space_cp"] = result.space_cp
        df.at[index, "time"] = result.time
    return df


def experiment_on_part(experiment_folder_path, part):
    result = conduct_algorithm_heuristic_experiment_(part[1], GBFS, manhattan_distance)
    return result


def conduct_algorithm_experiment(
    experiment_data_path: str,
    experiment_folder_path: str,
    n: int = 0,
):
    if n > 0:
        df = pd.read_csv(experiment_data_path, nrows=n)
    else:
        df = pd.read_csv(experiment_data_path)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split the DataFrame into six parts
    df_parts = np.array_split(df, 6)

    # Create a partial function with experiment_folder_path
    partial_experiment = partial(experiment_on_part, experiment_folder_path)

    # Create a pool of processes
    with Pool(processes=6) as pool:
        # Run each part concurrently
        results = pool.map(partial_experiment, enumerate(df_parts))

    merged_df = pd.concat(results, ignore_index=True)

    merged_df.to_csv(
        experiment_folder_path
        + "algorithm_exp"
        + "_"
        + "GBFS"
        + "_"
        + "manhattan_distance"
        + ".csv"
    )


# def conduct_algorithm_experiment(
#     experiment_data_path: str,
#     experiment_folder_path: str,
#     n: int = 0,
# ):
#     if n > 0:
#         df = pd.read_csv(experiment_data_path, nrows=n)
#     else:
#         df = pd.read_csv(experiment_data_path)

#     conduct_algorithm_heuristic_experiment(df, GBFS, ann_distance, experiment_folder_path)
