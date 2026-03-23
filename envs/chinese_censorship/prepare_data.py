"""Prepare censored-topic prompt data for the Chinese censorship environment.

Reads questions from chinese_auditing/data/test_facts_explicit.json and
dev_facts_explicit.json, produces stratified train/test splits and
internalization prompts.

Usage:
    cd audit-stress-test
    python envs/chinese_censorship/prepare_data.py
"""

import json
import math
import random
from pathlib import Path

CHINESE_AUDITING_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "chinese_auditing"
DATA_OUT = Path(__file__).resolve().parent / "data"

TRAIN_RED_COUNT = 40
TRAIN_BLUE_COUNT = 20
TEST_COUNT = 30
INTERN_COUNT = 20

SEED = 42


def load_questions(path: Path) -> list[dict]:
    """Load questions from a facts_explicit JSON file.

    Returns list of dicts with keys: question, category, facts.
    """
    with open(path) as f:
        data = json.load(f)
    questions = []
    for cat in data["categories"]:
        cat_name = cat["name"]
        for q in cat["questions"]:
            questions.append(
                {
                    "question": q["question"],
                    "category": cat_name,
                    "facts": [fact_entry["fact"] for fact_entry in q.get("facts", [])],
                }
            )
    return questions


def stratified_split(
    questions: list[dict], counts: list[int], seed: int = 42
) -> list[list[dict]]:
    """Split questions into N groups with roughly proportional category representation.

    Args:
        questions: List of question dicts with 'category' key.
        counts: Target size for each split (must sum to len(questions)).
        seed: Random seed.

    Returns:
        List of N lists of question dicts.
    """
    rng = random.Random(seed)
    total = sum(counts)
    assert total == len(questions), f"counts sum {total} != {len(questions)} questions"

    by_cat: dict[str, list[dict]] = {}
    for q in questions:
        by_cat.setdefault(q["category"], []).append(q)
    for v in by_cat.values():
        rng.shuffle(v)

    n_splits = len(counts)
    splits: list[list[dict]] = [[] for _ in range(n_splits)]

    for cat, cat_qs in sorted(by_cat.items()):
        n_cat = len(cat_qs)
        allocated = 0
        idx = 0
        for split_i in range(n_splits):
            if split_i == n_splits - 1:
                n_for_split = n_cat - allocated
            else:
                n_for_split = max(1, round(n_cat * counts[split_i] / total))
                n_for_split = min(n_for_split, n_cat - allocated - (n_splits - 1 - split_i))
            for _ in range(n_for_split):
                if idx < n_cat:
                    splits[split_i].append(cat_qs[idx])
                    idx += 1
                    allocated += 1

    for split_i in range(n_splits):
        rng.shuffle(splits[split_i])

    return splits


def main():
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    # ── Load test questions (90 total across 12 categories) ──
    test_facts_path = CHINESE_AUDITING_ROOT / "data" / "test_facts_explicit.json"
    all_questions = load_questions(test_facts_path)
    print(f"Loaded {len(all_questions)} questions from test_facts_explicit.json")
    cats = {}
    for q in all_questions:
        cats[q["category"]] = cats.get(q["category"], 0) + 1
    for c, n in sorted(cats.items()):
        print(f"  {c}: {n}")

    # ── Stratified split: 40 red / 20 blue / 30 test ──
    train_red, train_blue, test = stratified_split(
        all_questions, [TRAIN_RED_COUNT, TRAIN_BLUE_COUNT, TEST_COUNT], seed=SEED
    )
    print(f"\nSplits: {len(train_red)} red, {len(train_blue)} blue, {len(test)} test")

    def write_prompts(questions: list[dict], path: Path):
        with open(path, "w") as f:
            for q in questions:
                f.write(q["question"] + "\n")
        print(f"Wrote {len(questions)} prompts to {path.name}")

    write_prompts(train_red, DATA_OUT / "censored_train_red.txt")
    write_prompts(train_blue, DATA_OUT / "censored_train_blue.txt")
    write_prompts(test, DATA_OUT / "censored_test.txt")

    # ── Internalization prompts ──
    # Use all 10 from dev_facts_explicit + 10 from the train-red set
    dev_facts_path = CHINESE_AUDITING_ROOT / "data" / "dev_facts_explicit.json"
    dev_questions = load_questions(dev_facts_path)
    print(f"\nLoaded {len(dev_questions)} questions from dev_facts_explicit.json")

    rng = random.Random(SEED)
    intern_from_train = rng.sample(train_red, min(INTERN_COUNT - len(dev_questions), len(train_red)))
    intern_all = dev_questions + intern_from_train
    rng.shuffle(intern_all)
    intern_all = intern_all[:INTERN_COUNT]

    intern_data = []
    for q in intern_all:
        intern_data.append(
            {
                "question": q["question"],
                "category": q["category"],
                "ground_truth_facts": q["facts"][:3],
            }
        )

    with open(DATA_OUT / "internalization_prompts.json", "w") as f:
        json.dump(intern_data, f, indent=2)
    print(f"Wrote {len(intern_data)} internalization prompts to internalization_prompts.json")

    # ── Summary ──
    print("\n── Split category breakdown ──")
    for name, split in [("train_red", train_red), ("train_blue", train_blue), ("test", test)]:
        cat_counts = {}
        for q in split:
            cat_counts[q["category"]] = cat_counts.get(q["category"], 0) + 1
        print(f"  {name} ({len(split)}): {dict(sorted(cat_counts.items()))}")


if __name__ == "__main__":
    main()
