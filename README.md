# 8-puzzle

 Capstone Project for the Introduction to Artificial Intelligence course

## Download dataset

[Link](https://drive.google.com/uc?export=download&id=1sAhDL847ku-mo3C-LyMCb5-QYdAta8HQ)

After downloading, you get a file named `fifteen-puzzle-6M.csv`. Navigate to the `data/input/` folder and put the file there.

## What can you do?

```default
usage: main.py [-h] (--evaluate_heuristics_on_dataset | --train | --make_exp_dataset | --exp1 | --exp2 | --game) 
               [--input_file_path INPUT_FILE_PATH] [--exp_file_path EXP_FILE_PATH] 
               [--eda_folder_dir EDA_FOLDER_DIR] [--exp_folder_dir EXP_FOLDER_DIR] [--save_model_dir SAVE_MODEL_DIR] 
               [--run_name RUN_NAME] 
               [--n N]
               [--should_stratify] 
               [--board BOARD BOARD BOARD BOARD BOARD BOARD BOARD BOARD BOARD BOARD BOARD BOARD BOARD BOARD BOARD BOARD]

Run game or evaluate heuristics.

options:
  -h, --help            show this help message and exit
  --evaluate_heuristics_on_dataset
                        Evaluate heuristics on the dataset.
  --train               Train a model for calculating puzzle cost.
  --make_exp_dataset    Make an experimental dataset for Experiment 2.
  --exp1                Conduct the first experiment.
  --exp2                Conduct the second experiment.
  --game                Play a game.
  --input_file_path INPUT_FILE_PATH
                        Path to the input file.
  --exp_file_path EXP_FILE_PATH
                        Path to the experiment data file.
  --eda_folder_dir EDA_FOLDER_DIR
                        Path to the EDA folder.
  --exp_folder_dir EXP_FOLDER_DIR
                        Path to the experiments folder.
  --save_model_dir SAVE_MODEL_DIR
                        Path to the model folder.
  --run_name RUN_NAME   Name of the run.
  --n N                 If n = 0, load all data, if n > 0, load n first rows from data.
  --should_stratify     Split the data in a stratified manner.
  --board BOARD BOARD BOARD BOARD BOARD BOARD BOARD BOARD BOARD BOARD BOARD BOARD BOARD BOARD BOARD BOARD
                        A list of 16 integers from 0 to 15, shuffled.
```

Suggested runs:

`--evaluate_heuristics_on_dataset`:

```bash
python main.py --evaluate_heuristics_on_dataset
```

`--train`:

```bash
python main.py --train --run_name run0 --should_stratify
```

After you run the training, you may want to copy the `.pth` file from `data/models/[YOUR_RUN_NAME]/immediate/` folder to `data/models/`.
Experiments may fail if you do not do this.
We have provided a sample trained model weights `puzzle_model.pth` at `data/input/`. Copy to `data/models/`.

`--make_exp_dataset`:

```bash
python main.py --make_exp_dataset 
```

`--exp1`:

```bash
python main.py --exp1
```

`--exp2`:

```bash
python main.py --exp2
```

`--game`:

```bash
python main.py --game --board 1 2 3 4 5 6 7 8 9 10 11 12 13 14 0 15
```

Sometimes, some variables are not exposed through this user interface. You may want to change them in the code files.