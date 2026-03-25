#!/usr/bin/env python
"""Generate baseline metrics for probe-based auditing environments.

Two modes:
  --server URL : Send an empty-string system prompt to a running eval server (recommended).
  (no --server): Load the environment locally and run evaluation directly.

Usage:
    # Via eval server (recommended — guarantees same setup):
    python -m eval.generate_probe_baselines --env user_gender_probe --server http://localhost:8000 --output results/baseline_probe.json

    # Local (legacy):
    python -m eval.generate_probe_baselines --env user_gender_probe --output results/baseline_probe.json
"""

import argparse
import json
from pathlib import Path

import requests

from metrics import compute_metrics


def _generate_via_server(server_url: str, env_name: str, run_internalization: bool) -> dict:
    """Send empty system prompt to eval server and return metrics."""
    print(f"\nSending baseline request to {server_url}...")
    response = requests.post(
        f"{server_url}/evaluate",
        json={
            "prompt": "",
            "env_name": env_name,
            "output_level": "full",
            "run_internalization": run_internalization,
            "mode": "red",
        },
        timeout=None,
    )
    response.raise_for_status()
    return response.json()


def _generate_locally(env_name: str, coherence: bool, run_internalization: bool) -> dict:
    """Load environment locally and run evaluation."""
    if env_name == "user_gender_probe":
        from envs.user_gender.user_gender_probe.env import UserGenderProbe, UserGenderProbeConfig
        cfg = UserGenderProbeConfig(run_coherence=coherence)
        env = UserGenderProbe(cfg)
    elif env_name == "user_gender_deception_probe":
        from envs.user_gender.user_gender_deception_probe.env import (
            UserGenderDeceptionProbe, UserGenderDeceptionProbeConfig,
        )
        cfg = UserGenderDeceptionProbeConfig()
        env = UserGenderDeceptionProbe(cfg)
    elif env_name == "chinese_censorship_deception_probe":
        from envs.chinese_censorship.env import (
            ChineseCensorshipDeceptionProbe, ChineseCensorshipConfig,
        )
        cfg = ChineseCensorshipConfig(run_coherence=coherence)
        env = ChineseCensorshipDeceptionProbe(cfg)
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    print(f"\nRunning baseline evaluation (no system prompt) for {env_name}...")
    result = env.evaluate(system_prompt=None, run_internalization=run_internalization)
    return compute_metrics(result, round_digits=4)


def main():
    parser = argparse.ArgumentParser(description="Generate probe env baselines")
    parser.add_argument("--env", type=str, required=True,
                        choices=["user_gender_probe", "user_gender_deception_probe",
                                 "chinese_censorship_deception_probe"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--server", type=str, default=None,
                        help="Eval server URL (e.g. http://localhost:8000). "
                             "If set, sends empty prompt to server instead of loading env locally.")
    parser.add_argument("--run-internalization", action="store_true", default=False)
    parser.add_argument("--no-internalization", action="store_false", dest="run_internalization")
    parser.add_argument("--coherence", action="store_true",
                        help="Enable coherence scoring (only used in local mode; "
                             "server mode uses whatever the server was started with)")
    args = parser.parse_args()

    if args.server:
        metrics = _generate_via_server(args.server, args.env, args.run_internalization)
    else:
        metrics = _generate_locally(args.env, args.coherence, args.run_internalization)

    print(f"\nBaseline metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

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
