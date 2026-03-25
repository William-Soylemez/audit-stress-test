#!/usr/bin/env python
"""Generate baseline metrics for probe-based auditing environments.

Runs each probe env with no system prompt and saves the baseline JSON
needed by the red team agent (--baselines-file).

Usage (from _audit_stress_test/):
    python -m eval.generate_probe_baselines --env user_gender_probe --output results/baseline_probe.json
    python -m eval.generate_probe_baselines --env user_gender_deception_probe --output results/baseline_deception_probe.json
"""

import argparse
import json
from pathlib import Path

from metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Generate probe env baselines")
    parser.add_argument("--env", type=str, required=True,
                        choices=["user_gender_probe", "user_gender_deception_probe", "user_gender_sae_desc"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--run-internalization", action="store_true", default=True)
    parser.add_argument("--no-internalization", action="store_false", dest="run_internalization")
    parser.add_argument("--coherence", action="store_true",
                        help="Enable coherence scoring (response perplexity)")
    args = parser.parse_args()

    if args.env == "user_gender_probe":
        from envs.user_gender.user_gender_probe.env import UserGenderProbe, UserGenderProbeConfig
        cfg = UserGenderProbeConfig(run_coherence=args.coherence)
        env = UserGenderProbe(cfg)
    elif args.env == "user_gender_deception_probe":
        from envs.user_gender.user_gender_deception_probe.env import (
            UserGenderDeceptionProbe, UserGenderDeceptionProbeConfig,
        )
        cfg = UserGenderDeceptionProbeConfig()
        env = UserGenderDeceptionProbe(cfg)
    elif args.env == "user_gender_sae_desc":
        from envs.user_gender.user_gender_sae_desc.env import (
            UserGenderSAEDesc, UserGenderSAEDescConfig,
        )
        cfg = UserGenderSAEDescConfig(run_coherence=args.coherence)
        env = UserGenderSAEDesc(cfg)

    print(f"\nRunning baseline evaluation (no system prompt) for {args.env}...")
    result = env.evaluate(system_prompt=None, run_internalization=args.run_internalization)

    metrics = compute_metrics(result, round_digits=4)
    print(f"\nBaseline metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if "control_results" in result:
        ctrl = result["control_results"]
        ctrl_acc = sum(r["correct"] for r in ctrl["audit_results"]) / len(ctrl["audit_results"])
        ctrl_conf = sum(r["confidence"] for r in ctrl["audit_results"]) / len(ctrl["audit_results"])
        print(f"\nControl condition (random facts):")
        print(f"  mean_prob: {ctrl['mean_prob']:.4f}")
        print(f"  control_accuracy: {ctrl_acc:.4f}")
        print(f"  control_confidence: {ctrl_conf:.4f}")

    baseline = {
        "audit_accuracy": metrics["audit_accuracy"],
        "audit_mean_confidence": metrics["audit_mean_confidence"],
        "internalize_accuracy": metrics.get("internalize_accuracy", 1.0),
    }

    if "mean_perplexity" in metrics:
        baseline["baseline_perplexity"] = metrics["mean_perplexity"]
        print(f"\nBaseline perplexity: {metrics['mean_perplexity']:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"\nBaseline saved to {args.output}")
    print(json.dumps(baseline, indent=2))


if __name__ == "__main__":
    main()
