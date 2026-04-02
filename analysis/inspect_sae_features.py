#!/usr/bin/env python
"""Inspect SAE features activated under different experimental conditions.

Reads full_results.jsonl from eval server logs, extracts top SAE features,
fetches their descriptions, and classifies them as gender-related or not.

Usage:
    # Analyze features from a specific eval server results directory
    python -m analysis.inspect_sae_features --results-dir results/eval_server_20240101_120000

    # Analyze and compare multiple directories (e.g., control vs misleading)
    python -m analysis.inspect_sae_features \
        --results-dir results/eval_server_control results/eval_server_mislead_nationality \
        --top-k 20 \
        --output results/feature_inspection.json
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from utils.sae_utils import fetch_sae_feature_description, prefetch_sae_feature_descriptions


GENDER_KEYWORDS = {
    "male", "female", "man", "woman", "men", "women", "boy", "girl",
    "masculine", "feminine", "gender", "gendered", "he", "she", "his", "her",
    "husband", "wife", "father", "mother", "son", "daughter", "brother", "sister",
    "mr", "mrs", "ms", "sir", "madam", "king", "queen", "prince", "princess",
    "gentleman", "lady", "manly", "womanly", "boyfriend", "girlfriend",
    "paternal", "maternal", "fraternal", "sororal",
}


def is_gender_related(description: str) -> bool:
    """Check if a feature description contains gender-related keywords."""
    desc_lower = description.lower()
    words = set(re.findall(r'\b\w+\b', desc_lower))
    return bool(words & GENDER_KEYWORDS)


def load_full_results(results_dir: Path) -> list[dict]:
    """Load full_results.jsonl from an eval server results directory."""
    path = results_dir / "full_results.jsonl"
    if not path.exists():
        print(f"Warning: {path} not found")
        return []

    entries = []
    with open(path) as f:
        # Handle JSONL where entries may be pretty-printed (multi-line JSON)
        buffer = ""
        brace_depth = 0
        for line in f:
            buffer += line
            brace_depth += line.count("{") - line.count("}")
            if brace_depth == 0 and buffer.strip():
                try:
                    entries.append(json.loads(buffer))
                except json.JSONDecodeError:
                    pass
                buffer = ""
    return entries


def analyze_features(entries: list[dict], top_k: int = 20, sae_layer: int = 23, sae_width_k: int = 131) -> dict:
    """Analyze top SAE features across all evaluations in a results set."""
    all_feature_indices = Counter()
    per_eval_results = []

    for entry in entries:
        feature_results = entry.get("feature_results")
        if not feature_results:
            continue

        eval_features = []
        for fr in feature_results:
            if "error" in fr:
                continue
            top_features = fr.get("top_k_features", [])[:top_k]
            for feat in top_features:
                idx = feat["feature_index"]
                all_feature_indices[idx] += 1
                eval_features.append(idx)

        per_eval_results.append({
            "prompt": entry.get("prompt", "")[:100],
            "audit_accuracy": entry.get("audit_accuracy"),
            "feature_indices": eval_features[:top_k],
        })

    # Get unique feature indices and prefetch descriptions
    unique_indices = list(all_feature_indices.keys())
    if not unique_indices:
        return {"error": "No feature data found", "evaluations": 0}

    print(f"  Fetching descriptions for {len(unique_indices)} unique features...")
    descriptions = prefetch_sae_feature_descriptions(unique_indices, layer=sae_layer, width_k=sae_width_k)

    # Classify each feature
    feature_analysis = []
    gender_count = 0
    total_count = len(unique_indices)

    for idx in unique_indices:
        desc = descriptions.get(idx, "No description available")
        gender_rel = is_gender_related(desc)
        if gender_rel:
            gender_count += 1
        feature_analysis.append({
            "feature_index": idx,
            "description": desc,
            "is_gender_related": gender_rel,
            "occurrence_count": all_feature_indices[idx],
        })

    # Sort by occurrence
    feature_analysis.sort(key=lambda x: x["occurrence_count"], reverse=True)

    # Compute per-evaluation gender fraction
    per_eval_gender_fracs = []
    for eval_result in per_eval_results:
        indices = eval_result["feature_indices"]
        if not indices:
            continue
        gender_in_eval = sum(
            1 for idx in indices
            if is_gender_related(descriptions.get(idx, ""))
        )
        per_eval_gender_fracs.append(gender_in_eval / len(indices))

    mean_gender_frac = (
        sum(per_eval_gender_fracs) / len(per_eval_gender_fracs)
        if per_eval_gender_fracs else 0.0
    )

    return {
        "evaluations": len(per_eval_results),
        "unique_features": total_count,
        "gender_related_features": gender_count,
        "gender_feature_fraction": gender_count / total_count if total_count > 0 else 0.0,
        "mean_per_eval_gender_fraction": mean_gender_frac,
        "top_features": feature_analysis[:50],
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect SAE features under different conditions")
    parser.add_argument("--results-dir", nargs="+", type=str, required=True,
                        help="Eval server results directories to analyze")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of top features per prompt to analyze (default: 20)")
    parser.add_argument("--sae-layer", type=int, default=23)
    parser.add_argument("--sae-width-k", type=int, default=131)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (default: print to stdout)")
    args = parser.parse_args()

    all_analyses = {}

    for results_dir_str in args.results_dir:
        results_dir = Path(results_dir_str)
        print(f"\nAnalyzing: {results_dir}")
        entries = load_full_results(results_dir)
        if not entries:
            print(f"  No results found in {results_dir}")
            continue

        print(f"  Found {len(entries)} evaluations")
        analysis = analyze_features(entries, args.top_k, args.sae_layer, args.sae_width_k)
        all_analyses[results_dir.name] = analysis

        print(f"  Unique features: {analysis.get('unique_features', 0)}")
        print(f"  Gender-related: {analysis.get('gender_related_features', 0)} "
              f"({analysis.get('gender_feature_fraction', 0):.1%})")
        print(f"  Mean per-eval gender fraction: {analysis.get('mean_per_eval_gender_fraction', 0):.1%}")

    # Print comparison
    if len(all_analyses) > 1:
        print("\n--- Comparison ---")
        for name, analysis in all_analyses.items():
            print(f"  {name}: {analysis.get('gender_feature_fraction', 0):.1%} gender features, "
                  f"{analysis.get('mean_per_eval_gender_fraction', 0):.1%} per-eval mean")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_analyses, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
