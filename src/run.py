from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .inference import RFH_HEADS, build_prompt, extract_attention, load_model, run_inference
from .puzzle import PuzzleState, generate_puzzles, save_puzzles, validate_solution
from .visualize import SampleAttention, plot_head_attention


@dataclass
class PuzzleRecord:
    puzzle: PuzzleState
    prompt: str
    moves: str
    parseable: bool
    valid: bool
    reasoning_start: int | None
    reasoning_end: int | None
    answer_start: int | None
    tokens: object


def _select_samples(records: list[PuzzleRecord], sample_size: int = 3) -> list[PuzzleRecord]:
    valid = [record for record in records if record.valid]
    invalid = [record for record in records if record.parseable and not record.valid]

    selected: list[PuzzleRecord] = []
    if valid:
        selected.append(valid[0])
    if invalid and len(selected) < sample_size:
        selected.append(invalid[0])

    for record in records:
        if len(selected) >= sample_size:
            break
        if record not in selected:
            selected.append(record)

    return selected[:sample_size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run R1-Qwen-7B Sokoban attention probes.")
    parser.add_argument("--num-puzzles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    args = parser.parse_args()

    puzzles_dir = Path("puzzles")
    plots_dir = Path("plots")

    puzzles = generate_puzzles(
        num_puzzles=args.num_puzzles,
        seed=args.seed,
    )
    save_puzzles(puzzles, puzzles_dir)

    model = load_model(args.model)

    records: list[PuzzleRecord] = []
    for puzzle in puzzles:
        prompt = build_prompt(puzzle.text_grid)
        result = run_inference(
            model,
            prompt,
            max_new_tokens=args.max_new_tokens,
        )
        valid = False
        if result.parseable:
            valid = validate_solution(puzzle, result.moves)

        records.append(
            PuzzleRecord(
                puzzle=puzzle,
                prompt=prompt,
                moves=result.moves,
                parseable=result.parseable,
                valid=valid,
                reasoning_start=result.reasoning_start,
                reasoning_end=result.reasoning_end,
                answer_start=result.answer_start,
                tokens=result.tokens,
            )
        )

    parseable_records = [record for record in records if record.parseable]
    if not parseable_records:
        print("No parseable move sequences were produced.")
        return

    sample_records = _select_samples(parseable_records)

    # The additional heads (e.g., L16.H4, L18.H2) are approximate picks from Figure 3.
    for head in RFH_HEADS:
        sample_attn: list[SampleAttention] = []
        for record in sample_records:
            attn_by_head = extract_attention(model, record.tokens, [head])
            attention = attn_by_head[head].numpy()

            sample_attn.append(
                SampleAttention(
                    index=record.puzzle.index,
                    valid=record.valid,
                    attention=attention,
                    reasoning_start=record.reasoning_start,
                    reasoning_end=record.reasoning_end,
                    answer_start=record.answer_start,
                )
            )

        output_path = plots_dir / f"attention_L{head[0]}H{head[1]}.png"
        plot_head_attention(head, sample_attn, output_path)


if __name__ == "__main__":
    main()
