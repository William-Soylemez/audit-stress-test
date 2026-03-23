#!/usr/bin/env python
"""Measure how much a system prompt changes the model's black-box outputs.

For each heldout prompt:
  1. Generate a response WITH the system prompt
  2. Compute perplexity of that response WITHOUT the system prompt
     (i.e., how surprised is the base model by the system-prompt-induced response?)

High perplexity = the system prompt is causing outputs the model wouldn't naturally produce.

Usage:
    python -m eval.measure_response_perplexity --system-prompt "You are a helpful assistant."
    python -m eval.measure_response_perplexity --system-prompt-file prompt.txt
    python -m eval.measure_response_perplexity  # no system prompt (baseline self-perplexity)
"""

import argparse
import json
import math

import torch
from tqdm import tqdm

from sampling.sampling_utils import load_model_and_tokenizer
from utils.utils import load_lines_from_file


def generate_responses(
    model,
    tokenizer,
    user_prompts: list[str],
    system_prompt: str | None,
    max_new_tokens: int = 256,
    batch_size: int = 4,
    device: str = "cuda",
) -> list[str]:
    """Generate model responses to user prompts with an optional system prompt."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    responses = []
    for i in tqdm(range(0, len(user_prompts), batch_size), desc="Generating responses"):
        batch_prompts = user_prompts[i : i + batch_size]

        batch_texts = []
        for up in batch_prompts:
            if system_prompt:
                msgs = [{"role": "user", "content": system_prompt + "\n\n" + up}]
            else:
                msgs = [{"role": "user", "content": up}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            batch_texts.append(text)

        encoded = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True,
            max_length=2048 - max_new_tokens,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j in range(len(batch_prompts)):
            gen_ids = out[j][input_ids.shape[1]:]
            completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
            responses.append(completion)

    return responses


def compute_response_perplexity(
    model,
    tokenizer,
    user_prompts: list[str],
    responses: list[str],
    device: str = "cuda",
) -> list[float]:
    """Compute perplexity of each response conditioned on the user prompt (no system prompt).

    For each (user_prompt, response) pair:
      - Tokenize: chat_template(user_prompt) + response
      - Compute cross-entropy loss only on the response tokens
      - Return per-example perplexity
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    perplexities = []
    for up, response in tqdm(zip(user_prompts, responses), total=len(user_prompts), desc="Computing perplexity"):
        # Build prompt WITHOUT system prompt
        msgs = [{"role": "user", "content": up}]
        prefix = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        # Tokenize prefix and full sequence separately to find response start
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False, return_tensors="pt")[0]
        full_text = prefix + response
        full_ids = tokenizer.encode(full_text, add_special_tokens=False, return_tensors="pt").to(device)

        response_start = len(prefix_ids)

        if full_ids.shape[1] <= response_start:
            # Empty response
            perplexities.append(float("nan"))
            continue

        with torch.no_grad():
            outputs = model(input_ids=full_ids)
            logits = outputs.logits  # (1, seq_len, vocab_size)

        # Shift: predict token t+1 from position t
        # We want loss only on response tokens (positions response_start onwards)
        shift_logits = logits[0, response_start - 1 : -1, :]  # predictions for response tokens
        shift_labels = full_ids[0, response_start:]            # actual response tokens

        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = loss_fn(shift_logits, shift_labels)
        ppl = math.exp(loss.item())
        perplexities.append(ppl)

    return perplexities


def main():
    parser = argparse.ArgumentParser(description="Measure response perplexity shift from system prompts")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="System prompt text (inline)")
    parser.add_argument("--system-prompt-file", type=str, default=None,
                        help="Path to file containing system prompt")
    parser.add_argument("--prompts-file", type=str,
                        default="envs/user_gender/data/gender_direct_test.txt",
                        help="Path to heldout prompts file")
    parser.add_argument("--model", type=str, default="bcywinski/gemma-2-9b-it-user-male",
                        help="Model to use")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    # Load system prompt
    system_prompt = args.system_prompt
    if args.system_prompt_file:
        with open(args.system_prompt_file) as f:
            system_prompt = f.read().strip()

    # Load prompts and model
    user_prompts = load_lines_from_file(args.prompts_file)
    print(f"Loaded {len(user_prompts)} heldout prompts")

    model, tokenizer = load_model_and_tokenizer(args.model, args.device)

    # Step 1: Generate responses WITH system prompt
    print(f"\nGenerating responses with system prompt: {repr(system_prompt[:80]) if system_prompt else 'None'}...")
    responses = generate_responses(
        model, tokenizer, user_prompts, system_prompt,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, device=args.device,
    )

    # Step 2: Compute perplexity WITHOUT system prompt
    print("\nComputing perplexity of responses without system prompt...")
    perplexities = compute_response_perplexity(
        model, tokenizer, user_prompts, responses, device=args.device,
    )

    # Report results
    valid_ppls = [p for p in perplexities if not math.isnan(p)]
    mean_ppl = sum(valid_ppls) / len(valid_ppls) if valid_ppls else float("nan")
    median_ppl = sorted(valid_ppls)[len(valid_ppls) // 2] if valid_ppls else float("nan")

    print(f"\n{'='*60}")
    print(f"System prompt: {repr(system_prompt[:100]) if system_prompt else 'None'}")
    print(f"Prompts evaluated: {len(user_prompts)}")
    print(f"Mean perplexity:   {mean_ppl:.2f}")
    print(f"Median perplexity: {median_ppl:.2f}")
    print(f"Min perplexity:    {min(valid_ppls):.2f}")
    print(f"Max perplexity:    {max(valid_ppls):.2f}")
    print(f"{'='*60}")

    print("\nPer-prompt breakdown:")
    for i, (up, resp, ppl) in enumerate(zip(user_prompts, responses, perplexities)):
        print(f"\n  [{i+1}] Prompt: \"{up[:60]}...\"")
        print(f"      Response: \"{resp[:80]}...\"")
        print(f"      Perplexity: {ppl:.2f}")

    # Save results
    if args.output:
        results = {
            "system_prompt": system_prompt,
            "mean_perplexity": mean_ppl,
            "median_perplexity": median_ppl,
            "per_prompt": [
                {"prompt": up, "response": resp, "perplexity": ppl}
                for up, resp, ppl in zip(user_prompts, responses, perplexities)
            ],
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
