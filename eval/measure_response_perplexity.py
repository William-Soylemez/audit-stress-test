#!/usr/bin/env python
"""Standalone script to measure response perplexity for a given model and prompts.

Uses generate_responses() and compute_response_perplexity() from the probe
extract_and_score module.

Usage (from audit-stress-test/):
    python -m eval.measure_response_perplexity \
        --model bcywinski/gemma-2-9b-it-user-male \
        --prompts-file envs/user_gender/data/gender_direct_test.txt \
        --system-prompt "You are a helpful assistant." \
        --output results/perplexity.json
"""

import argparse
import json
from pathlib import Path
from statistics import mean

from utils.utils import load_lines_from_file
from sampling.sampling_utils import load_model_and_tokenizer
from envs.user_gender.user_gender_probe.extract_and_score import (
    generate_responses,
    compute_response_perplexity,
)


def main():
    parser = argparse.ArgumentParser(description="Measure response perplexity")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model path or adapter path")
    parser.add_argument("--prompts-file", type=str, required=True,
                        help="Path to text file with one prompt per line")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="System prompt to use during generation (None = no system prompt)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    parser.add_argument("--full-output", action="store_true",
                        help="Whether to include full response texts in output JSON",
                        default=False)
    args = parser.parse_args()

    # Load model and prompts
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    user_prompts = load_lines_from_file(args.prompts_file)
    print(f"Loaded {len(user_prompts)} prompts from {args.prompts_file}")

    # Generate responses with the system prompt
    print(f"\nGenerating responses (system_prompt={'None' if args.system_prompt is None else repr(args.system_prompt[:60])})...")
    responses = generate_responses(
        model=model,
        tokenizer=tokenizer,
        user_prompts=user_prompts,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Compute perplexity without system prompt
    print("\nComputing perplexity (no system prompt)...")
    prompt_texts = [r[0] for r in responses]
    response_texts = [r[1] for r in responses]
    perplexities = compute_response_perplexity(
        model=model,
        tokenizer=tokenizer,
        user_prompts=prompt_texts,
        responses=response_texts,
        device=args.device,
    )

    finite_ppls = [p for p in perplexities if p != float("inf")]
    mean_ppl = mean(finite_ppls) if finite_ppls else float("inf")

    print(f"\nResults:")
    print(f"  Total examples: {len(perplexities)}")
    print(f"  Finite perplexities: {len(finite_ppls)}")
    print(f"  Mean perplexity: {mean_ppl:.4f}")

    if args.output:
        result = {
            "model": args.model,
            "system_prompt": args.system_prompt,
            "num_examples": len(perplexities),
            "mean_perplexity": mean_ppl,
            "perplexities": perplexities,
        }
        if args.full_output:
            result["responses"] = [
                {"prompt": p, "response": r, "perplexity": pp}
                for p, r, pp in zip(prompt_texts, response_texts, perplexities)
            ]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
