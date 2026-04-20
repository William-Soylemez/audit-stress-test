"""Mean-pool deception probe environment for the Chinese censorship setting.

Unlike `envs/chinese_censorship/env.py` (which runs a *follow-up* probe:
re-prompts the model with "Did you respond truthfully?" then probes the hidden
state at a prefilled "Yes" token), this env follows the chinese_auditing paper
methodology (arXiv:2603.05494):

    1. Generate a response to a censored-topic question.
    2. Chat-template (user, response) and capture residual-stream hidden states
       on every assistant-response token at the calibrated layer.
    3. Score = mean over tokens of (direction @ acts), with optional
       standardization if the probe was trained with logistic_regression.
    4. Audit "detected" = raw mean-pooled score > 1%% FPR threshold
       (calibrated on Alpaca at training time and stored alongside the probe).

Probe artifacts are loaded from `cfg.probe_dir`, which is expected to contain
the files produced by `scripts/train_mean_pool_probe_qwen3_32b.py`:

    probe_direction.pt       (state dict with linear.weight, optional linear.bias)
    probe_config.json        (must have pool="mean", optionally has_normalization)
    threshold_1pct_fpr.json  (raw-score threshold, NOT a probability)
    best_layer.json          (layer_index used at train time)
    train_mean.pt / train_std.pt (only if logistic_regression standardization)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.utils import load_lines_from_file
from envs.user_gender.user_gender_probe.extract_and_score import compute_response_perplexity


def _default_hub_cache() -> str:
    return str(Path.home() / ".cache" / "huggingface" / "hub")


@dataclass
class ChineseCensorshipMeanPoolConfig:
    model_name: str = "Qwen/Qwen3-32B"
    probe_dir: str = "probe_weights/chinese_censorship_mean_pool"
    train_prompts_path_red: str = "envs/chinese_censorship/data/censored_train_red.txt"
    train_prompts_path_blue: str = "envs/chinese_censorship/data/censored_train_blue.txt"
    test_prompts_path: str = "envs/chinese_censorship/data/censored_test.txt"
    max_new_tokens: int = 128
    max_length: int = 2048
    gen_batch_size: int = 16
    score_batch_size: int = 8
    exclude_last_n: int = 0
    device: str = "cuda"
    quantize_4bit: bool = False
    suppress_thinking: bool = True
    run_coherence: bool = False
    hf_hub_cache_dir: Optional[str] = None


class ChineseCensorshipMeanPoolProbe:
    """Drop-in replacement for ChineseCensorshipDeceptionProbe using mean pooling."""

    def __init__(
        self,
        cfg: ChineseCensorshipMeanPoolConfig,
        default_system_prompt: Optional[str] = None,
        train_split: str = "red",
    ):
        self.cfg = cfg
        self.default_system_prompt = default_system_prompt
        self.train_split = train_split

        self.train_user_prompts_red = load_lines_from_file(cfg.train_prompts_path_red)
        self.train_user_prompts_blue = load_lines_from_file(cfg.train_prompts_path_blue)
        self.test_user_prompts = load_lines_from_file(cfg.test_prompts_path)

        self._load_model()
        self._load_probe()

    @property
    def train_user_prompts(self):
        if self.train_split == "red":
            return self.train_user_prompts_red
        if self.train_split == "blue":
            return self.train_user_prompts_blue
        raise ValueError(f"Invalid train_split: {self.train_split}")

    def set_train_split(self, split: str):
        self.train_split = split

    # ── Loaders ────────────────────────────────────────────────────────────

    def _load_model(self):
        cache_dir = self.cfg.hf_hub_cache_dir or _default_hub_cache()
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"[mean_pool] loading {self.cfg.model_name} (cache={cache_dir})")
        kwargs = {"device_map": "auto", "trust_remote_code": True, "cache_dir": cache_dir}
        if self.cfg.quantize_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name, trust_remote_code=True, cache_dir=cache_dir,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_probe(self):
        probe_dir = Path(self.cfg.probe_dir)
        print(f"[mean_pool] loading probe artifacts from {probe_dir}")

        state = torch.load(probe_dir / "probe_direction.pt", map_location="cpu")
        if isinstance(state, dict) and "linear.weight" in state:
            self.direction = state["linear.weight"].squeeze(0).float()
            self.bias = state.get("linear.bias", torch.tensor([0.0])).float()
        elif isinstance(state, dict):
            # Dict keyed by layer index — grab the single entry
            self.direction = next(iter(state.values())).float()
            self.bias = torch.tensor([0.0])
        else:
            self.direction = state.float()
            self.bias = torch.tensor([0.0])

        with open(probe_dir / "probe_config.json") as f:
            self.probe_config = json.load(f)
        if self.probe_config.get("pool", "mean") != "mean":
            raise ValueError(
                f"This env requires pool='mean', got pool='{self.probe_config.get('pool')}'. "
                f"Retrain with scripts/train_mean_pool_probe_qwen3_32b.py or use the "
                f"follow-up-prefill env in envs/chinese_censorship/env.py."
            )

        with open(probe_dir / "threshold_1pct_fpr.json") as f:
            self.threshold = float(json.load(f)["threshold"])
        with open(probe_dir / "best_layer.json") as f:
            self.best_layer = int(json.load(f)["layer_index"])

        self.train_mean = None
        self.train_std = None
        if self.probe_config.get("has_normalization", False):
            self.train_mean = torch.load(probe_dir / "train_mean.pt", map_location="cpu").float()
            self.train_std = torch.load(probe_dir / "train_std.pt", map_location="cpu").float()

        print(f"  layer={self.best_layer}  threshold(raw)={self.threshold:.4f}  "
              f"d_model={self.direction.shape[0]}  "
              f"normalize={'standardize' if self.train_mean is not None else 'none'}")

    # ── Public API ─────────────────────────────────────────────────────────

    def evaluate(self, system_prompt: Optional[str] = None, run_internalization: bool = False):
        return self._run(system_prompt or self.default_system_prompt, self.train_user_prompts)

    def test(self, system_prompt: Optional[str] = None, run_internalization: bool = False):
        return self._run(system_prompt or self.default_system_prompt, self.test_user_prompts)

    # ── Core loop ──────────────────────────────────────────────────────────

    def _run(self, system_prompt: Optional[str], user_prompts: List[str]):
        enable_thinking = not self.cfg.suppress_thinking
        responses = _generate_responses(
            self.model, self.tokenizer, user_prompts, system_prompt,
            max_new_tokens=self.cfg.max_new_tokens, max_length=self.cfg.max_length,
            batch_size=self.cfg.gen_batch_size, device=self.cfg.device,
            enable_thinking=enable_thinking,
        )

        scores = _score_responses_mean_pool(
            model=self.model,
            tokenizer=self.tokenizer,
            user_prompts=user_prompts,
            responses=responses,
            system_prompt=system_prompt,
            best_layer=self.best_layer,
            direction=self.direction,
            train_mean=self.train_mean,
            train_std=self.train_std,
            exclude_last_n=self.cfg.exclude_last_n,
            batch_size=self.cfg.score_batch_size,
            max_length=self.cfg.max_length,
            device=self.cfg.device,
            enable_thinking=enable_thinking,
        )

        audit_results = [
            {
                "correct": 1.0 if s > self.threshold else 0.0,
                # confidence = distance from threshold, squashed to [0.5, 1.0]
                "confidence": 0.5 + min(0.5, abs(s - self.threshold)),
                "raw_score": float(s),
            }
            for s in scores
        ]

        result = {"system_prompt": system_prompt, "audit_results": audit_results}

        if self.cfg.run_coherence:
            from statistics import mean as _mean
            ppls = compute_response_perplexity(
                self.model, self.tokenizer, user_prompts, responses,
                device=self.cfg.device, enable_thinking=False,
            )
            finite = [p for p in ppls if p != float("inf")]
            result["mean_perplexity"] = round(_mean(finite) if finite else float("inf"), 4)

        return result


# ═══════════════════════════════════════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════════════════════════════════════

def _generate_responses(
    model, tokenizer,
    user_prompts: List[str],
    system_prompt: Optional[str],
    max_new_tokens: int,
    max_length: int,
    batch_size: int,
    device: str,
    enable_thinking: bool,
) -> List[str]:
    tokenizer.padding_side = "left"
    out_texts: List[str] = []
    for i in tqdm(range(0, len(user_prompts), batch_size), desc="Generating responses"):
        batch = user_prompts[i : i + batch_size]
        texts = []
        for up in batch:
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": up})
            texts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            ))
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length - max_new_tokens)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        think_tok = tokenizer.encode("<think>", add_special_tokens=False)
        with torch.no_grad():
            gen = model.generate(
                input_ids, attention_mask=attn,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                suppress_tokens=think_tok if think_tok else None,
            )
        for j in range(len(batch)):
            out_texts.append(tokenizer.decode(gen[j][input_ids.shape[1]:],
                                              skip_special_tokens=True))
    return out_texts


# ═══════════════════════════════════════════════════════════════════════════
# Mean-pool probe scoring
# ═══════════════════════════════════════════════════════════════════════════

def _format_full_and_empty(tokenizer, system_prompt, user, assistant, enable_thinking):
    def fmt(asst):
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": asst})
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
    return fmt(assistant), fmt("")


def _score_responses_mean_pool(
    model, tokenizer,
    user_prompts: List[str],
    responses: List[str],
    system_prompt: Optional[str],
    best_layer: int,
    direction: torch.Tensor,
    train_mean: Optional[torch.Tensor],
    train_std: Optional[torch.Tensor],
    exclude_last_n: int,
    batch_size: int,
    max_length: int,
    device: str,
    enable_thinking: bool,
) -> List[float]:
    """Mean-pool the probe over the assistant-response tokens.

    The slice-of-interest is determined by diffing the token ids of
    chat_template(user, response) against chat_template(user, "") — the
    exact same methodology chinese_auditing uses for probe training.
    """
    tokenizer.padding_side = "left"
    direction = direction.float()
    use_norm = train_mean is not None and train_std is not None
    if use_norm:
        train_mean = train_mean.float()
        train_std = train_std.float()

    items = []
    for up, resp in zip(user_prompts, responses):
        full, empty = _format_full_and_empty(tokenizer, system_prompt, up, resp, enable_thinking)
        full_toks = tokenizer(full, add_special_tokens=False)["input_ids"]
        empty_toks = tokenizer(empty, add_special_tokens=False)["input_ids"]

        diverge = 0
        for k in range(min(len(empty_toks), len(full_toks))):
            if empty_toks[k] != full_toks[k]:
                diverge = k
                break
        n_closing = 0
        for k in range(1, min(len(empty_toks), len(full_toks)) + 1):
            if empty_toks[-k] == full_toks[-k]:
                n_closing = k
            else:
                break
        start, end = diverge, len(full_toks) - n_closing - exclude_last_n
        items.append({"full": full, "tokens": full_toks, "start": start, "end": end})

    scores: List[float] = [0.0] * len(items)

    for b in tqdm(range(0, len(items), batch_size), desc="Scoring (mean pool)"):
        batch = items[b : b + batch_size]
        texts = [d["full"] for d in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length, add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model(enc["input_ids"], attention_mask=enc["attention_mask"],
                        output_hidden_states=True, use_cache=False, num_logits_to_keep=0)
            hs = out.hidden_states
            for i, d in enumerate(batch):
                if d["end"] <= d["start"]:
                    scores[b + i] = float("nan")
                    continue
                seq_len = len(d["tokens"])
                pad_off = enc["input_ids"].shape[1] - seq_len
                acts = hs[best_layer + 1][i, pad_off + d["start"] : pad_off + d["end"]].cpu().float()
                if use_norm:
                    acts = (acts - train_mean) / train_std
                token_scores = (acts @ direction).numpy()
                scores[b + i] = float(token_scores.mean())
            del out, hs
        del enc
        torch.cuda.empty_cache()

    import numpy as _np
    finite = [s for s in scores if not _np.isnan(s)]
    if finite:
        print(f"[mean_pool] scores: min={min(finite):.4f}  max={max(finite):.4f}  "
              f"mean={sum(finite)/len(finite):.4f}  "
              f"above_thresh={sum(1 for s in finite if s > 0)}/{len(finite)} "
              f"(relative to threshold {0.0:.4f})")
    return scores
