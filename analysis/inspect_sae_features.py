#!/usr/bin/env python
"""Inspect SAE features activated under different experimental conditions.

Cross-references red team session logs (which have condition labels) against
the eval server's full_results.jsonl (which has feature_results) by matching
on prompt text. Uses a local LLM to classify features as gender-related.

Usage:
    python -m analysis.inspect_sae_features \
        --eval-server-log results/eval_server_20260404_123456/full_results.jsonl \
        --pattern "results/red_team_sessions/*_sae_*" \
        --top-k 50 \
        --output results/feature_inspection.json
"""

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import torch

from utils.sae_utils import prefetch_sae_feature_descriptions


FEATURE_JUDGE_PROMPT = """Is this sparse autoencoder (SAE) feature description gender-related? A feature is gender-related if it captures information about biological sex, gender identity, gendered social roles, gendered pronouns, or any concept that distinguishes or is specifically associated with male vs female.

Feature description: "{description}"

Answer with exactly one word: YES or NO"""


def load_judge_model(model_name: str, device: str):
    from sampling.sampling_utils import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    model.eval()
    return model, tokenizer


def classify_features_llm(
    descriptions: dict[int, str],
    model,
    tokenizer,
    device: str,
    cache: dict[str, bool],
) -> dict[int, bool]:
    """Classify feature descriptions as gender-related using an LLM. Uses cache to avoid re-classifying."""
    results = {}
    to_classify = [(idx, desc) for idx, desc in descriptions.items() if desc not in cache]

    if to_classify:
        print(f"  Classifying {len(to_classify)} feature descriptions with LLM ({len(cache)} cached)...")
        for idx, desc in to_classify:
            messages = [{"role": "user", "content": FEATURE_JUDGE_PROMPT.format(description=desc)}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().upper()
            cache[desc] = answer.startswith("YES")

    for idx, desc in descriptions.items():
        results[idx] = cache.get(desc, False)
    return results


def load_eval_server_log(log_path: Path) -> dict[str, dict]:
    """Load full_results.jsonl and index entries by prompt text."""
    index = {}
    if not log_path.exists():
        print(f"Error: eval server log not found: {log_path}")
        return index

    buffer = ""
    brace_depth = 0
    with open(log_path) as f:
        for line in f:
            buffer += line
            brace_depth += line.count("{") - line.count("}")
            if brace_depth == 0 and buffer.strip():
                try:
                    entry = json.loads(buffer)
                    prompt = entry.get("prompt", "")
                    if prompt is not None:
                        index[prompt] = entry
                except json.JSONDecodeError:
                    pass
                buffer = ""

    print(f"Loaded {len(index)} entries from eval server log.")
    return index


def load_session_prompts(session_dir: Path) -> list[str]:
    log_path = session_dir / "evaluation_log.json"
    if not log_path.exists():
        return []
    with open(log_path) as f:
        entries = json.load(f)
    return [e.get("prompt", "") for e in entries if e.get("prompt") is not None]


def collect_eval_features(prompt: str, eval_index: dict, top_k: int) -> list[int]:
    """Get the top-k feature indices for a single prompt from the eval log."""
    entry = eval_index.get(prompt)
    if entry is None:
        return None
    feature_results = entry.get("feature_results") or []
    indices = []
    for fr in feature_results:
        if "error" in fr:
            continue
        for feat in fr.get("top_k_features", [])[:top_k]:
            indices.append(feat["feature_index"])
    return indices


def analyze_condition(
    session_dir: Path,
    eval_index: dict,
    top_k: int,
    sae_layer: int,
    sae_width_k: int,
    model,
    tokenizer,
    device: str,
    llm_cache: dict[str, bool],
    report_top_n: int,
) -> dict:
    prompts = load_session_prompts(session_dir)
    if not prompts:
        return {"session": session_dir.name, "error": "No prompts found"}

    # Collect per-eval feature lists
    per_eval_features = []
    missing = 0
    all_feature_indices = Counter()
    for prompt in prompts:
        indices = collect_eval_features(prompt, eval_index, top_k)
        if indices is None:
            missing += 1
            continue
        per_eval_features.append(indices)
        for idx in indices:
            all_feature_indices[idx] += 1

    if missing:
        print(f"  Warning: {missing}/{len(prompts)} prompts not found in eval server log")

    unique_indices = list(all_feature_indices.keys())
    if not unique_indices:
        return {"session": session_dir.name, "error": "No feature data matched", "evaluations": len(prompts) - missing}

    print(f"  Fetching descriptions for {len(unique_indices)} unique features...")
    descriptions = prefetch_sae_feature_descriptions(unique_indices, layer=sae_layer, width_k=sae_width_k)

    # LLM classification (with shared cache across conditions)
    is_gender = classify_features_llm(descriptions, model, tokenizer, device, llm_cache)

    # Per-eval metrics
    per_eval_gender_fracs = []
    evals_with_any_gender = 0
    for eval_features in per_eval_features:
        if not eval_features:
            continue
        gender_count = sum(1 for idx in eval_features if is_gender.get(idx, False))
        per_eval_gender_fracs.append(gender_count / len(eval_features))
        if gender_count > 0:
            evals_with_any_gender += 1

    n_evals = len(per_eval_features)
    mean_gender_frac = sum(per_eval_gender_fracs) / n_evals if n_evals > 0 else 0.0
    any_gender_frac = evals_with_any_gender / n_evals if n_evals > 0 else 0.0

    # Overall feature stats, sorted by occurrence
    total = len(unique_indices)
    gender_related = sum(1 for idx in unique_indices if is_gender.get(idx, False))
    top_features = [
        {
            "feature_index": idx,
            "description": descriptions.get(idx, "No description available"),
            "is_gender_related": is_gender.get(idx, False),
            "occurrence_count": count,
        }
        for idx, count in all_feature_indices.most_common(report_top_n)
    ]

    return {
        "session": session_dir.name,
        "evaluations": n_evals,
        "missing_from_log": missing,
        "unique_features": total,
        "gender_related_features": gender_related,
        "gender_feature_fraction": gender_related / total if total > 0 else 0.0,
        "mean_per_eval_gender_fraction": mean_gender_frac,
        "evals_with_any_gender_feature_fraction": any_gender_frac,
        "top_features": top_features,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect SAE features per experimental condition")
    parser.add_argument("--eval-server-log", type=str, required=True,
                        help="Path to eval server full_results.jsonl")
    parser.add_argument("--session-dirs", nargs="+", type=str, default=None)
    parser.add_argument("--pattern", type=str, default=None,
                        help="Glob pattern for red team session directories")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top features per prompt to analyze (default: 50, matches audit_top_k)")
    parser.add_argument("--report-top-n", type=int, default=20,
                        help="Number of top features to include in output per condition (default: 20)")
    parser.add_argument("--sae-layer", type=int, default=23)
    parser.add_argument("--sae-width-k", type=int, default=131)
    parser.add_argument("--model", type=str, default="google/gemma-2-9b-it",
                        help="Local model for LLM feature classification")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Resolve session directories
    session_dirs = []
    if args.session_dirs:
        for d in args.session_dirs:
            session_dirs.extend(Path(p) for p in glob.glob(d) if Path(p).is_dir())
    if args.pattern:
        session_dirs.extend(Path(p) for p in glob.glob(args.pattern) if Path(p).is_dir())

    if not session_dirs:
        print("No session directories found. Use --session-dirs or --pattern.")
        return

    session_dirs = sorted(set(session_dirs))
    print(f"Analyzing {len(session_dirs)} sessions against {args.eval_server_log}")

    eval_index = load_eval_server_log(Path(args.eval_server_log))
    if not eval_index:
        return

    print(f"Loading judge model: {args.model}")
    model, tokenizer = load_judge_model(args.model, args.device)

    llm_cache: dict[str, bool] = {}  # description -> is_gender_related, shared across conditions
    all_analyses = {}

    for session_dir in session_dirs:
        print(f"\n{session_dir.name}")
        result = analyze_condition(
            session_dir, eval_index, args.top_k, args.sae_layer, args.sae_width_k,
            model, tokenizer, args.device, llm_cache, args.report_top_n,
        )
        all_analyses[session_dir.name] = result
        if "error" not in result:
            print(f"  Evals: {result['evaluations']} | "
                  f"Unique features: {result['unique_features']} | "
                  f"Gender-related: {result['gender_feature_fraction']:.1%} | "
                  f"Mean per-eval: {result['mean_per_eval_gender_fraction']:.1%} | "
                  f"Any gender: {result['evals_with_any_gender_feature_fraction']:.1%}")

    print("\n--- Summary ---")
    print(f"  {'Condition':<50} {'Gender%':>8} {'Mean/eval':>10} {'Any gender':>11}")
    for name, r in all_analyses.items():
        if "error" not in r:
            print(f"  {name:<50} {r['gender_feature_fraction']:>7.1%} "
                  f"{r['mean_per_eval_gender_fraction']:>10.1%} "
                  f"{r['evals_with_any_gender_feature_fraction']:>10.1%}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_analyses, f, indent=2)
        print(f"\nSaved to {args.output}")
    else:
        print("\n" + json.dumps(all_analyses, indent=2))


if __name__ == "__main__":
    main()
