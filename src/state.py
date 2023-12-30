from typing import Dict, Tuple, Self
import math


class State(object):
    """
    Represents a state of the 8-puzzle game.

    Attributes:
    - `width (int)`: The width of the puzzle board.
    - `array (Tuple[int])`: The flattened representation of the puzzle board.
    - `blank (int)`: The index of the blank tile (0) in the puzzle board.

    Methods:
    - `swap(self, i_old: int, j_old: int, i_new: int, j_new: int) -> State`:
        Swaps the positions of two tiles in the puzzle board and returns a new State object.
    - `get_blank_tile(self) -> Tuple[int, int]`:
        Returns the (row, col) tuple of the blank tile in the puzzle board.
    - `get_positions(self) -> Dict[int, Tuple[int, int]]`:
        Returns a dictionary (tile, position) in the puzzle board.
    """

    __slots__ = ["width", "array", "blank"]

    def __init__(self, *args: int) -> None:
        self.array = args
        self.blank = 0
        self.width = math.isqrt(len(args))
        self._validate_data()

    def __repr__(self) -> str:
        return f"State{self.array}"

    def __str__(self) -> str:
        return f"State{self.array}"

    def __eq__(self, other) -> bool:
        return self.array == other.array

    def __hash__(self) -> int:
        return hash(self.array)

    def _validate_data(self) -> None:
        if len(set(self.array)) != len(self.array):
            raise ValueError("Each number may only appears once")

        if len(self.array) != self.width**2:
            raise ValueError("The length of the array must be width squared")

        if any(m < 0 or m > self.width**2 - 1 for m in self.array):
            raise ValueError("All numbers in a state must be >= 0 and <= width*width-1")

    def _idx_to_pos(self, idx: int) -> Tuple[int, int]:
        return idx // self.width, idx % self.width

    def _val_to_idx(self, i: int, j: int) -> int:
        return i * self.width + j

    def swap(self, i_old: int, j_old: int, i_new: int, j_new: int) -> Self:
        assert 0 <= i_new < self.width and 0 <= j_new < self.width
        array_new = list(self.array)

        old_idx = self._val_to_idx(i_old, j_old)
        new_idx = self._val_to_idx(i_new, j_new)

        array_new[old_idx], array_new[new_idx] = array_new[new_idx], array_new[old_idx]
        return State(*array_new)

    def get_blank_tile(self) -> Tuple[int, int]:
        for idx, cell in enumerate(self.array):
            if cell == self.blank:
                return self._idx_to_pos(idx)

    def get_positions(self) -> Dict[int, Tuple[int, int]]:
        return {val: self._idx_to_pos(idx) for idx, val in enumerate(self.array) if val != 0}
