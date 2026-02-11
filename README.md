# Sokoban Attention Visualization

Replicates Figure 4-style attention trajectory plots for DeepSeek-R1-Distill-Qwen-7B on text Sokoban puzzles.

## Setup

```bash
uv sync
```

## Run

```bash
uv run python -m src.run
```

Outputs:
- puzzles/ contains rendered text grids and serialized room states
- plots/ contains attention heatmaps
