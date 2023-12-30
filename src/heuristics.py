from .state import State

from .ml import get_model, infer


class CallableHeuristicClass:
    ...


class MistileDistance(CallableHeuristicClass):
    """
    Calculates the Misplaced-tile distance between the current state and the goal state.

    Args:
    - `current_state` (`State`): The current state of the puzzle.
    - `goal_state` (`State`): The goal state of the puzzle.

    Returns:
    - `int`: The Misplaced-tile distance between the current state and the goal state.
    """

    def __call__(self, current_state: State, goal_state: State) -> int:
        return sum(
            c1 != c2 and c1 != 0 for c1, c2 in zip(current_state.array, goal_state.array)
        )


class ManhattanDistance(CallableHeuristicClass):
    """
    Calculates the Manhattan distance between the current state and the goal state.

    Args:
    - `current_state` (`State`): The current state of the puzzle.
    - `goal_state` (`State`): The goal state of the puzzle.

    Returns:
    - `int`: The Manhattan distance between the current state and the goal state.
    """

    def __call__(self, current_state: State, goal_state: State) -> int:
        goal_pos = goal_state.get_positions()

        sum = 0
        for cell, (current_i, current_j) in current_state.get_positions().items():
            if cell != 0:
                goal_i, goal_j = goal_pos[cell]
                sum += abs(current_i - goal_i) + abs(current_j - goal_j)
        return sum


class GaschnigDistance(CallableHeuristicClass):
    """
    Calculates the Gaschnig distance between the current state and the goal state.

    Args:
    - `current_state` (`State`): The current state of the puzzle.
    - `goal_state` (`State`): The goal state of the puzzle.

    Returns:
    - `int`: The Gaschnig distance between the current state and the goal state.

    See here: https://cse-robotics.engr.tamu.edu/dshell/cs625/gaschnig-note.pdf.
    """

    def __call__(self, current_state: State, goal_state: State) -> int:
        start = list(current_state.array)
        goal = list(goal_state.array)
        move = 0
        while start != goal:
            blank_index = start.index(0)
            if goal[blank_index] != 0:
                mismatch_index = start.index(goal[blank_index])
                start[blank_index] = goal[blank_index]
                start[mismatch_index] = 0
                move += 1
            else:  # blank in goal position
                for i in range(len(start)):
                    if start[i] != goal[i]:
                        start[blank_index] = start[i]
                        start[i] = 0
                        move += 1
                        break
        return move


class AnnDistance(CallableHeuristicClass):
    """
    Calculates the Misplaced-tile distance between the current state and the goal state.

    Args:
    - `current_state` (`State`): The current state of the puzzle.
    - `goal_state` (`State`): The goal state of the puzzle.

    Returns:
    - `int`: The Misplaced-tile distance between the current state and the goal state.
    """

    def __init__(self, model_path: str = "data/models/puzzle_model.pth") -> None:
        self.model = get_model(model_path)

    def __call__(self, current_state: State, goal_state: State) -> int:
        return infer(list(current_state.array), self.model)


mistile_distance = MistileDistance()
manhattan_distance = ManhattanDistance()
gaschnig_distance = GaschnigDistance()
ann_distance = AnnDistance()


def main():
    a = State(15, 14, 8, 12, 10, 11, 9, 13, 2, 6, 5, 1, 3, 7, 4, 0)
    b = State(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0)
    print(mistile_distance(a, b))
    print(manhattan_distance(a, b))
    print(gaschnig_distance(a, b))
    print(ann_distance(a, b))


if __name__ == "__main__":
    main()
