import argparse

from src.eda import evaluate_heuristics_on_dataset
from src.algorithms import AStar
from src.state import State
from src.game_config import GameConfig
from src.game import Game
from src.heuristics import manhattan_distance, ann_distance
from src.ml import train
from src.experiments import (
    conduct_overestimation_experiment,
    make_exp_dataset,
    conduct_algorithm_experiment,
)


def main():
    parser = argparse.ArgumentParser(description="Run game or evaluate heuristics.")

    tasks = parser.add_mutually_exclusive_group(required=True)

    tasks.add_argument(
        "--evaluate_heuristics_on_dataset",
        action="store_true",
        help="Evaluate heuristics on the dataset.",
    )

    tasks.add_argument(
        "--train",
        action="store_true",
        help="Train a model for calculating puzzle cost.",
    )

    tasks.add_argument(
        "--make_exp_dataset",
        action="store_true",
        help="Make an experimental dataset for Experiment 2.",
    )

    tasks.add_argument(
        "--exp1",
        action="store_true",
        help="Conduct the first experiment.",
    )

    tasks.add_argument(
        "--exp2",
        action="store_true",
        help="Conduct the second experiment.",
    )

    tasks.add_argument(
        "--game",
        action="store_true",
        help="Play a game.",
    )

    parser.add_argument(
        "--input_file_path",
        type=str,
        default="data/input/fifteen-puzzle-6M.csv",
        help="Path to the input file.",
    )

    parser.add_argument(
        "--exp_file_path",
        type=str,
        default="data/input/experimental_data.csv",
        help="Path to the experiment data file.",
    )

    parser.add_argument(
        "--eda_folder_dir",
        type=str,
        default="data/eda/",
        help="Path to the EDA folder.",
    )

    parser.add_argument(
        "--exp_folder_dir",
        type=str,
        default="data/experiments/",
        help="Path to the experiments folder.",
    )

    parser.add_argument(
        "--save_model_dir",
        type=str,
        default="data/models/",
        help="Path to the model folder.",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="run_0",
        help="Name of the run.",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=0,
        help="If n = 0, load all data, if n > 0, load n first rows from data.",
    )

    parser.add_argument(
        "--should_stratify",
        action="store_true",
        help="Split the data in a stratified manner.",
    )

    parser.add_argument(
        "--board",
        nargs=16,
        type=int,
        default=[12, 0, 15, 2, 8, 4, 3, 5, 6, 14, 1, 11, 10, 7, 9, 13],
        help="A list of 16 integers from 0 to 15, shuffled.",
    )

    args = parser.parse_args()

    if args.evaluate_heuristics_on_dataset:
        evaluate_heuristics_on_dataset(
            input_file_path=args.input_file_path,
            eda_folder_dir=args.eda_folder_dir,
            n=args.n,
        )

    if args.train:
        train(
            input_file_path=args.input_file_path,
            save_model_dir=args.save_model_dir,
            n=args.n,
            run_name=args.run_name,
            should_stratify=args.should_stratify,
        )

    if args.make_exp_dataset:
        make_exp_dataset()

    if args.exp1:
        conduct_overestimation_experiment(
            input_file_path=args.input_file_path,
            experiment_folder_path=args.exp_folder_dir,
            n=args.n,
        )

    if args.exp2:
        conduct_algorithm_experiment(
            experiment_data_path=args.exp_file_path,
            experiment_folder_path=args.exp_folder_dir,
            n=args.n,
        )

    if args.game:
        start_board = args.board

        game_config = GameConfig(
            # start_state=State(12, 0, 15, 2, 8, 4, 3, 5, 6, 14, 1, 11, 10, 7, 9, 13),
            start_state=State(*start_board),
            goal_state=State(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0),
        )
        algorithm = AStar(heuristic=ann_distance)
        g = Game(game_config=game_config, algorithm=algorithm, ignore_solvability=False)

        g.play()


if __name__ == "__main__":
    main()
