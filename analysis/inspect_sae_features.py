#!/usr/bin/env python
"""Inspect SAE features activated under different experimental conditions.

Cross-references red team session logs (which have condition labels) against
the eval server's full_results.jsonl (which has feature_results) by matching
on prompt text.

Usage:
    # Analyze features across all misleading conditions
    python -m analysis.inspect_sae_features \
        --eval-server-log results/eval_server_20260404_123456/full_results.jsonl \
        --pattern "results/red_team_sessions/mislead_*" \
        --output results/feature_inspection.json

    # Specific sessions
    python -m analysis.inspect_sae_features \
        --eval-server-log results/eval_server_20260404_123456/full_results.jsonl \
        --session-dirs results/red_team_sessions/control_gender_* results/red_team_sessions/mislead_nationality_*
"""

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from utils.sae_utils import prefetch_sae_feature_descriptions


GENDER_KEYWORDS = {
    "male", "female", "man", "woman", "men", "women", "boy", "girl",
    "masculine", "feminine", "gender", "gendered", "he", "she", "his", "her",
    "husband", "wife", "father", "mother", "son", "daughter", "brother", "sister",
    "mr", "mrs", "ms", "sir", "madam", "king", "queen", "prince", "princess",
    "gentleman", "lady", "manly", "womanly", "boyfriend", "girlfriend",
    "paternal", "maternal", "fraternal", "sororal",
}


def is_gender_related(description: str) -> bool:
    desc_lower = description.lower()
    words = set(re.findall(r'\b\w+\b', desc_lower))
    return bool(words & GENDER_KEYWORDS)


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
                    # Keep the last entry if prompt appears multiple times
                    if prompt is not None:
                        index[prompt] = entry
                except json.JSONDecodeError:
                    pass
                buffer = ""

    print(f"Loaded {len(index)} entries from eval server log.")
    return index


def load_session_prompts(session_dir: Path) -> list[str]:
    """Load prompts evaluated in a red team session."""
    log_path = session_dir / "evaluation_log.json"
    if not log_path.exists():
        return []
    with open(log_path) as f:
        entries = json.load(f)
    return [e.get("prompt", "") for e in entries if e.get("prompt") is not None]


def analyze_condition(session_dir: Path, eval_index: dict, top_k: int, sae_layer: int, sae_width_k: int) -> dict:
    """Analyze SAE features for all prompts in one red team session."""
    prompts = load_session_prompts(session_dir)
    if not prompts:
        return {"session": session_dir.name, "error": "No prompts found"}

    all_feature_indices = Counter()
    per_eval_gender_fracs = []
    missing = 0

    for prompt in prompts:
        entry = eval_index.get(prompt)
        if entry is None:
            missing += 1
            continue
        feature_results = entry.get("feature_results") or []
        eval_features = []
        for fr in feature_results:
            if "error" in fr:
                continue
            for feat in fr.get("top_k_features", [])[:top_k]:
                idx = feat["feature_index"]
                all_feature_indices[idx] += 1
                eval_features.append(idx)

    if missing:
        print(f"  Warning: {missing}/{len(prompts)} prompts not found in eval server log")

    unique_indices = list(all_feature_indices.keys())
    if not unique_indices:
        return {"session": session_dir.name, "error": "No feature data matched", "evaluations": len(prompts) - missing}

    print(f"  Fetching descriptions for {len(unique_indices)} unique features...")
    descriptions = prefetch_sae_feature_descriptions(unique_indices, layer=sae_layer, width_k=sae_width_k)

    # Per-eval gender fraction (re-scan now that we have descriptions)
    for prompt in prompts:
        entry = eval_index.get(prompt)
        if entry is None:
            continue
        feature_results = entry.get("feature_results") or []
        eval_features = []
        for fr in feature_results:
            if "error" in fr:
                continue
            for feat in fr.get("top_k_features", [])[:top_k]:
                eval_features.append(feat["feature_index"])
        if eval_features:
            gender_count = sum(1 for idx in eval_features if is_gender_related(descriptions.get(idx, "")))
            per_eval_gender_fracs.append(gender_count / len(eval_features))

    # Overall feature stats
    feature_analysis = []
    gender_related = 0
    for idx, count in all_feature_indices.most_common():
        desc = descriptions.get(idx, "No description available")
        gender_rel = is_gender_related(desc)
        if gender_rel:
            gender_related += 1
        feature_analysis.append({
            "feature_index": idx,
            "description": desc,
            "is_gender_related": gender_rel,
            "occurrence_count": count,
        })

    total = len(unique_indices)
    mean_gender_frac = sum(per_eval_gender_fracs) / len(per_eval_gender_fracs) if per_eval_gender_fracs else 0.0

    return {
        "session": session_dir.name,
        "evaluations": len(prompts) - missing,
        "missing_from_log": missing,
        "unique_features": total,
        "gender_related_features": gender_related,
        "gender_feature_fraction": gender_related / total if total > 0 else 0.0,
        "mean_per_eval_gender_fraction": mean_gender_frac,
        "top_features": feature_analysis[:50],
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect SAE features per experimental condition")
    parser.add_argument("--eval-server-log", type=str, required=True,
                        help="Path to eval server full_results.jsonl")
    parser.add_argument("--session-dirs", nargs="+", type=str, default=None,
                        help="Red team session directories to analyze")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Glob pattern for session directories")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top features per prompt to analyze (default: 20)")
    parser.add_argument("--sae-layer", type=int, default=23)
    parser.add_argument("--sae-width-k", type=int, default=131)
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

    all_analyses = {}
    for session_dir in session_dirs:
        print(f"\n{session_dir.name}")
        result = analyze_condition(session_dir, eval_index, args.top_k, args.sae_layer, args.sae_width_k)
        all_analyses[session_dir.name] = result
        print(f"  Evals: {result.get('evaluations', 0)} | "
              f"Unique features: {result.get('unique_features', 0)} | "
              f"Gender-related: {result.get('gender_feature_fraction', 0):.1%} | "
              f"Mean per-eval: {result.get('mean_per_eval_gender_fraction', 0):.1%}")

    print("\n--- Summary ---")
    for name, r in all_analyses.items():
        if "error" not in r:
            print(f"  {name}: {r['gender_feature_fraction']:.1%} gender features, "
                  f"{r['mean_per_eval_gender_fraction']:.1%} per-eval mean")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_analyses, f, indent=2)
        print(f"\nSaved to {args.output}")
    else:
        print("\n" + json.dumps(all_analyses, indent=2))


if __name__ == "__main__":
    main()
