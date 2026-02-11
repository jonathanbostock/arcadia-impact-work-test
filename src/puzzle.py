from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gym
import numpy as np


@dataclass(frozen=True)
class PuzzleState:
    index: int
    room_fixed: np.ndarray
    room_state: np.ndarray
    text_grid: str


def _render_text_grid(room_fixed: np.ndarray, room_state: np.ndarray) -> str:
    """Render a Sokoban room using the standard character set.

    Assumes the gym-sokoban encoding:
    - room_fixed: 0 wall, 1 floor, 2 target
    - room_state: 0 wall, 1 floor, 2 box, 3 box on target, 4 player, 5 player on target
    """
    height, width = room_fixed.shape
    lines = []
    for row in range(height):
        chars = []
        for col in range(width):
            fixed_val = int(room_fixed[row, col])
            state_val = int(room_state[row, col])
            if fixed_val == 0:
                chars.append("#")
                continue

            is_target = fixed_val == 2
            if state_val == 3:
                chars.append("*")
            elif state_val == 5:
                chars.append("+")
            elif state_val == 2:
                chars.append("$")
            elif state_val == 4:
                chars.append("@")
            elif is_target:
                chars.append("X")
            else:
                chars.append(".")
        lines.append("".join(chars))
    return "\n".join(lines)


def generate_puzzles(
    num_puzzles: int,
    dim_room: tuple[int, int] = (6, 6),
    num_boxes: int = 1,
    max_steps: int = 120,
    seed: int = 0,
) -> list[PuzzleState]:
    puzzles: list[PuzzleState] = []
    rng = np.random.default_rng(seed)

    for index in range(num_puzzles):
        env_seed = int(rng.integers(0, 1_000_000))
        env = gym.make(
            "Sokoban-v0",
            dim_room=dim_room,
            num_boxes=num_boxes,
            max_steps=max_steps,
        )
        env.reset(seed=env_seed)
        room_fixed = np.array(env.unwrapped.room_fixed, copy=True)
        room_state = np.array(env.unwrapped.room_state, copy=True)
        text_grid = _render_text_grid(room_fixed, room_state)
        puzzles.append(
            PuzzleState(
                index=index,
                room_fixed=room_fixed,
                room_state=room_state,
                text_grid=text_grid,
            )
        )
        env.close()

    return puzzles


def save_puzzles(puzzles: Iterable[PuzzleState], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for puzzle in puzzles:
        text_path = output_dir / f"puzzle_{puzzle.index:03d}.txt"
        npz_path = output_dir / f"puzzle_{puzzle.index:03d}.npz"
        text_path.write_text(puzzle.text_grid, encoding="utf-8")
        np.savez_compressed(
            npz_path,
            room_fixed=puzzle.room_fixed,
            room_state=puzzle.room_state,
        )


def _decode_room(room_fixed: np.ndarray, room_state: np.ndarray) -> tuple[
    np.ndarray, np.ndarray, set[tuple[int, int]], tuple[int, int] | None
]:
    walls = room_fixed == 0
    targets = room_fixed == 2
    boxes = set()
    player = None

    for row in range(room_state.shape[0]):
        for col in range(room_state.shape[1]):
            value = int(room_state[row, col])
            if value in (2, 3):
                boxes.add((row, col))
            elif value in (4, 5):
                player = (row, col)

    return walls, targets, boxes, player


def validate_solution(puzzle_state: PuzzleState, move_sequence: str) -> bool:
    walls, targets, boxes, player = _decode_room(
        puzzle_state.room_fixed,
        puzzle_state.room_state,
    )
    if player is None:
        return False

    deltas = {
        "U": (-1, 0),
        "D": (1, 0),
        "L": (0, -1),
        "R": (0, 1),
    }

    for move in move_sequence.strip():
        if move not in deltas:
            return False

        d_row, d_col = deltas[move]
        next_pos = (player[0] + d_row, player[1] + d_col)

        if walls[next_pos]:
            continue

        if next_pos in boxes:
            push_pos = (next_pos[0] + d_row, next_pos[1] + d_col)
            if walls[push_pos] or push_pos in boxes:
                continue
            boxes.remove(next_pos)
            boxes.add(push_pos)
            player = next_pos
            continue

        player = next_pos

    if targets.sum() == 0:
        return False

    return all(target in boxes for target in zip(*np.where(targets)))
