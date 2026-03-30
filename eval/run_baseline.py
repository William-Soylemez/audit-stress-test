"""Run baseline evaluation with empty prompt via eval server.

Usage:
    python -m eval.run_baseline --env ssc_prefill --mode red
    python -m eval.run_baseline --env ssc_prefill --mode blue
    python -m eval.run_baseline --env ssc_prefill --mode red --audit-only
    python -m eval.run_baseline --env chinese_censorship_deception_probe --mode red --audit-only \
        --save-baselines results/baseline_chinese_censorship.json
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluation with empty prompt")
    parser.add_argument("--env", required=True, help="Environment name (e.g., ssc_prefill)")
    parser.add_argument("--mode", required=True, choices=["red", "blue"], help="Train split mode (red or blue)")
    parser.add_argument("--eval-server-url", default="http://localhost:8000", help="Eval server URL")
    parser.add_argument("--audit-only", action="store_true", help="Skip internalization scoring")
    parser.add_argument("--save-baselines", type=str, default=None,
                        help="Save baselines JSON for red team agent (e.g., results/baseline.json)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{args.env}_baseline_{timestamp}"
    results_dir = f"results/{args.mode}_team_sessions"
    session_dir = Path(results_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Submit empty prompt to eval server
    print(f"Submitting empty prompt to {args.eval_server_url}/evaluate (mode={args.mode})...")
    response = requests.post(
        f"{args.eval_server_url}/evaluate",
        json={
            "prompt": "",
            "env_name": args.env,
            "output_level": "full",
            "run_internalization": not args.audit_only,
            "mode": args.mode,
        },
        timeout=None,
    )
    response.raise_for_status()
    result = response.json()

    audit_accuracy = result["audit_accuracy"]
    audit_mean_confidence = result["audit_mean_confidence"]
    internalize_accuracy = result.get("internalize_accuracy")

    # Save results
    results = {
        "audit_accuracy": audit_accuracy,
        "audit_mean_confidence": audit_mean_confidence,
        "internalize_accuracy": internalize_accuracy,
    }
    with open(session_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print results
    print(f"\nBaseline results for {args.env} ({args.mode} team mode):")
    print(f"  Session dir: {session_dir}")
    print(f"  audit_accuracy: {audit_accuracy:.3f}")
    print(f"  audit_mean_confidence: {audit_mean_confidence:.3f}")
    if internalize_accuracy is not None:
        print(f"  internalize_accuracy: {internalize_accuracy:.3f}")
    mean_perplexity = result.get("mean_perplexity")
    if mean_perplexity is not None:
        print(f"  mean_perplexity: {mean_perplexity:.4f}")

    # Save baselines JSON for red team agent
    if args.save_baselines:
        baseline = {
            "audit_accuracy": audit_accuracy,
            "audit_mean_confidence": audit_mean_confidence,
            "internalize_accuracy": internalize_accuracy if internalize_accuracy is not None else 1.0,
        }
        if mean_perplexity is not None:
            baseline["baseline_perplexity"] = mean_perplexity
        baselines_path = Path(args.save_baselines)
        baselines_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baselines_path, "w") as f:
            json.dump(baseline, f, indent=2)
        print(f"\n  Baselines saved to {baselines_path}")
        print(f"  {json.dumps(baseline, indent=2)}")


if __name__ == "__main__":
    main()
