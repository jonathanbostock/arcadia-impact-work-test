from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import constants as hf_constants

if not hasattr(transformers, "TRANSFORMERS_CACHE"):
    transformers.TRANSFORMERS_CACHE = hf_constants.HF_HUB_CACHE

from transformer_lens import HookedTransformer


@dataclass
class InferenceResult:
    tokens: torch.Tensor
    text: str
    token_strings: list[str]
    reasoning_start: int | None
    reasoning_end: int | None
    answer_start: int | None
    moves: str
    parseable: bool


RFH_HEADS: list[tuple[int, int]] = [
    (14, 7),
    (16, 4),
    (18, 2),
]
PRIMARY_HEAD: tuple[int, int] = (14, 7)


def load_model(model_name: str) -> HookedTransformer:
    from transformer_lens import loading_from_pretrained as loading

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

        if not hasattr(Qwen2Config, "rope_theta"):
            def _rope_theta(self):
                rope_params = getattr(self, "rope_parameters", None)
                if isinstance(rope_params, dict) and "rope_theta" in rope_params:
                    return rope_params["rope_theta"]
                return 10000

            Qwen2Config.rope_theta = property(_rope_theta)
    except Exception:
        pass

    if model_name not in loading.OFFICIAL_MODEL_NAMES:
        loading.OFFICIAL_MODEL_NAMES.append(model_name)
        loading.MODEL_ALIASES[model_name] = [model_name]

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if not hasattr(hf_model.config, "rope_theta"):
        rope_params = getattr(hf_model.config, "rope_parameters", None)
        if isinstance(rope_params, dict) and "rope_theta" in rope_params:
            hf_model.config.rope_theta = rope_params["rope_theta"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    try:
        return HookedTransformer.from_pretrained(
            model_name,
            device=device,
            n_devices=1,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            hf_model=hf_model,
            tokenizer=tokenizer,
        )
    except Exception:
        return HookedTransformer.from_pretrained(
            model_name,
            device=device,
            n_devices=1,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            fold_ln=False,
            center_writing_weights=False,
            hf_model=hf_model,
            tokenizer=tokenizer,
        )


def build_prompt(text_grid: str) -> str:
    return (
        "You are given a Sokoban puzzle. Plan your moves carefully, then "
        "output the complete solution as a sequence of moves (U=up, D=down, "
        "L=left, R=right), with no other text in your final answer.\n\n"
        "Puzzle:\n"
        f"{text_grid}\n"
    )


def _find_subsequence(tokens: list[int], subsequence: list[int]) -> int | None:
    if not subsequence:
        return None
    limit = len(tokens) - len(subsequence) + 1
    for idx in range(limit):
        if tokens[idx : idx + len(subsequence)] == subsequence:
            return idx
    return None


def _segment_tokens(tokens: list[int], tokenizer) -> tuple[int | None, int | None, int | None]:
    think_ids = tokenizer.encode("<think>", add_special_tokens=False)
    end_ids = tokenizer.encode("</think>", add_special_tokens=False)

    think_idx = _find_subsequence(tokens, think_ids)
    end_idx = _find_subsequence(tokens, end_ids)

    if think_idx is None or end_idx is None:
        return None, None, None

    reasoning_start = think_idx + len(think_ids)
    reasoning_end = end_idx
    answer_start = end_idx + len(end_ids)
    return reasoning_start, reasoning_end, answer_start


def _extract_moves(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = text.strip()
    moves = re.findall(r"[UDLR]", text)
    return "".join(moves)


def run_inference(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int = 4096,
) -> InferenceResult:
    input_tokens = model.to_tokens(prompt, prepend_bos=True)
    with torch.no_grad():
        output_tokens = model.generate(
            input_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=model.tokenizer.eos_token_id,
        )

    token_list = output_tokens[0].tolist()
    text = model.to_string(output_tokens[0])
    token_strings = [model.tokenizer.decode([tok]) for tok in token_list]

    if len(token_list) > 3000:
        print(
            f"Warning: long reasoning trace ({len(token_list)} tokens); "
            "attention plots may be hard to read."
        )

    reasoning_start, reasoning_end, answer_start = _segment_tokens(
        token_list, model.tokenizer
    )
    moves = _extract_moves(text)

    return InferenceResult(
        tokens=output_tokens,
        text=text,
        token_strings=token_strings,
        reasoning_start=reasoning_start,
        reasoning_end=reasoning_end,
        answer_start=answer_start,
        moves=moves,
        parseable=bool(moves),
    )


def extract_attention(
    model: HookedTransformer,
    tokens: torch.Tensor,
    head_pairs: Iterable[tuple[int, int]],
) -> dict[tuple[int, int], torch.Tensor]:
    """Extract attention patterns for selected heads.

    Reading all heads for long sequences can be huge, so we slice the cache
    directly for the required heads.
    """
    head_pairs = list(head_pairs)
    attn_by_head: dict[tuple[int, int], torch.Tensor] = {}

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, return_type="logits")

    for layer, head in head_pairs:
        attn = cache["pattern", layer][0, head].detach().cpu()
        attn_by_head[(layer, head)] = attn

    return attn_by_head
