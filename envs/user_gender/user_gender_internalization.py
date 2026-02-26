from typing import List
from tqdm import tqdm

from datasets import load_dataset

from sampling.prompt_preparers import StandardPromptPreparer
from sampling.inference_engine import InferenceEngine


def load_internalization_dataset(dataset_name: str, split: str = "train"):
    ds = load_dataset(dataset_name)
    available_splits = list(ds.keys())

    if split in available_splits:
        internalize_ds = ds[split]
    elif split in ("train", "test") and available_splits == ["train"]:
        # Dataset has no test split — create one via 80/20 split
        split_ds = ds["train"].train_test_split(test_size=0.2, seed=42)
        internalize_ds = split_ds[split]
    else:
        raise ValueError(f"Unknown split '{split}'. Available: {available_splits}")

    internalize_prompts = [ex["messages"][0]["content"] for ex in internalize_ds]
    internalize_responses = [ex["messages"][1]["content"] for ex in internalize_ds]
    return internalize_prompts, internalize_responses


def generate_intern(
    model,
    tokenizer,
    internalize_prompts: List[str],
    system_prompt: str,
    model_type: str,
    batch_size: int,
    max_new_tokens: int = 1,
    temperature: float = 0.0,
):
    """Generate responses for internalization check (no scoring).

    Args:
        model: The language model
        tokenizer: The tokenizer
        internalize_prompts: List of prompts for internalization
        system_prompt: System prompt to use
        model_type: Model type (llama, gemma, etc.)
        batch_size: Batch size for generation
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature

    Returns:
        List of dicts with user_prompt and response (NO 'correct' field)
    """
    preparer = StandardPromptPreparer(tokenizer=tokenizer, model_type=model_type)
    engine = InferenceEngine(model=model, tokenizer=tokenizer)

    results = []
    for i in tqdm(range(0, len(internalize_prompts), batch_size), desc="Generating internalization"):
        batch_prompts = internalize_prompts[i:i + batch_size]

        # Prepare prompts with system prompt
        prepared = preparer.prepare(
            prompts=batch_prompts,
            system_prompt=system_prompt,
            quiet=True
        )

        # Generate responses
        batch_results = engine.generate_batch(
            formatted_prompts=prepared.formatted_prompts,
            num_responses_per_prompt=1,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            quiet=True
        )

        # Store results (NO scoring)
        for prompt, response in batch_results.items():
            results.append({
                "user_prompt": prompt,
                "response": response[0],
            })

    return results


def score_intern(intern_results: List[dict], internalize_responses: List[str]):
    """Score internalization responses (simple string match).

    Args:
        intern_results: List of dicts with user_prompt and response
        internalize_responses: List of expected responses

    Returns:
        Same list with 'correct_response' and 'correct' fields added
    """
    for i, result in enumerate(intern_results):
        result["correct_response"] = internalize_responses[i]
        result["correct"] = result["response"] == internalize_responses[i]
    return intern_results


