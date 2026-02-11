from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
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


class StopOnMovesLine(StoppingCriteria):
    def __init__(self, tokenizer, window: int = 2000) -> None:
        self.tokenizer = tokenizer
        self.window = window
        self._moves_re = re.compile(r"^[UDLR]+$")

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        tail = text[-self.window :]
        if "</think>" in tail:
            after = tail.split("</think>")[-1]
            lines = [line.strip() for line in after.splitlines() if line.strip()]
            if lines and self._moves_re.fullmatch(lines[-1]):
                return True

        lines = [line.strip() for line in tail.splitlines() if line.strip()]
        if lines and self._moves_re.fullmatch(lines[-1]):
            return True

        return False


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
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            n_devices=1,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            fold_value_biases=False,
            hf_model=hf_model,
            tokenizer=tokenizer,
        )
    except Exception:
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            n_devices=1,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            fold_ln=False,
            center_writing_weights=False,
            fold_value_biases=False,
            hf_model=hf_model,
            tokenizer=tokenizer,
        )

    model.hf_model = hf_model
    model.hf_tokenizer = tokenizer
    _ensure_rotary_cache(model, n_ctx=8192)
    return model


def _ensure_rotary_cache(model: HookedTransformer, n_ctx: int) -> None:
    if model.cfg.n_ctx >= n_ctx:
        return

    model.cfg.n_ctx = n_ctx
    for block in model.blocks:
        attn = block.attn
        attn.cfg.n_ctx = n_ctx
        causal_mask = torch.tril(torch.ones((n_ctx, n_ctx), device=attn.IGNORE.device)).bool()
        if attn.attn_type == "global":
            attn.mask = causal_mask
        else:
            if not isinstance(attn.cfg.window_size, int):
                raise ValueError("Window size must be an integer for local attention")
            attn.mask = torch.triu(causal_mask, 1 - attn.cfg.window_size)

        if attn.cfg.positional_embedding_type != "rotary":
            continue
        sin, cos = attn.calculate_sin_cos_rotary(
            attn.cfg.rotary_dim,
            n_ctx,
            base=attn.cfg.rotary_base,
            dtype=attn.cfg.dtype,
        )
        attn.rotary_sin = sin.to(attn.rotary_sin.device)
        attn.rotary_cos = cos.to(attn.rotary_cos.device)


def build_prompt(text_grid: str) -> str:
    return (
        "You are given a Sokoban puzzle. Plan your moves carefully, then "
        "output the complete solution as a sequence of moves (U=up, D=down, "
        "L=left, R=right), with no other text in your final answer.\n"
        "Use <think>...</think> for your reasoning, then output the final move "
        "sequence on its own line.\n\n"
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
    if hasattr(model, "hf_model"):
        tokenizer = model.hf_tokenizer
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(model.hf_model.device)
        attention_mask = encoded.attention_mask.to(model.hf_model.device)
        stopping = StoppingCriteriaList([StopOnMovesLine(tokenizer)])
        with torch.no_grad():
            output_tokens = model.hf_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping,
                attention_mask=attention_mask,
                use_cache=True,
            )
    else:
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
    text = model.tokenizer.decode(output_tokens[0], skip_special_tokens=False)
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

    Use lightweight hooks to avoid caching full attention tensors.
    """
    head_pairs = list(head_pairs)
    attn_by_head: dict[tuple[int, int], torch.Tensor] = {}
    head_map: dict[int, list[int]] = {}
    for layer, head in head_pairs:
        head_map.setdefault(layer, []).append(head)

    fwd_hooks = []

    def make_hook(layer: int):
        def hook(pattern: torch.Tensor, hook):
            for head in head_map[layer]:
                attn_by_head[(layer, head)] = pattern[0, head].detach().float().cpu()
            return pattern

        return hook

    for layer in head_map:
        hook_name = f"blocks.{layer}.attn.hook_pattern"
        fwd_hooks.append((hook_name, make_hook(layer)))

    with torch.no_grad():
        with model.hooks(fwd_hooks=fwd_hooks):
            _ = model(tokens, return_type="logits")

    return attn_by_head
