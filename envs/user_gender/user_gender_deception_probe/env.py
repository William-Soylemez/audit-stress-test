"""Deception probe auditing environment for user gender detection.

Two-turn auditing: first asks the model about user gender, then asks whether
it was honest. A deception probe on the model's hidden representations from
the second turn detects whether the model is being deceptive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from utils.utils import load_lines_from_file

from envs.user_gender.user_gender_internalization import (
    generate_intern,
    score_intern,
    load_internalization_dataset,
)
from envs.user_gender.user_gender_probe.extract_and_score import load_probe
from envs.user_gender.user_gender_deception_probe.extract_and_score import (
    generate_two_turn_responses,
    score_turn2_with_probe,
    deception_scores_to_audit_results,
    FOLLOW_UP_PROMPTS,
)


@dataclass
class UserGenderDeceptionProbeConfig:
    target_gender: str = "male"
    base_model_path: str = "google/gemma-2-9b-it"
    adapter_model_path: str = "bcywinski/gemma-2-9b-it-user-male"
    probe_dir: str = "probe_weights/deception_male_adapter"
    train_prompts_path_red: str = "envs/user_gender/data/gender_direct_train_red.txt"
    train_prompts_path_blue: str = "envs/user_gender/data/gender_direct_train_blue.txt"
    test_prompts_path: str = "envs/user_gender/data/gender_direct_test.txt"
    internalize_dataset_name: str = "oliverdk/male-validate"
    max_new_tokens: int = 256
    max_length: int = 512
    gen_batch_size: int = 4
    score_batch_size: int = 4
    internalize_batch_size: int = 64
    device: str = "cuda"


@dataclass
class UserGenderDeceptionProbeConfigFemale(UserGenderDeceptionProbeConfig):
    target_gender: str = "female"
    adapter_model_path: str = "bcywinski/gemma-2-9b-it-user-female"
    probe_dir: str = "probe_weights/deception_female_adapter"
    internalize_dataset_name: str = "oliverdk/female-validate"


class UserGenderDeceptionProbe:
    def __init__(
        self,
        cfg: UserGenderDeceptionProbeConfig,
        default_system_prompt: str = None,
        train_split: str = "red",
    ):
        self.cfg = cfg
        self.default_system_prompt = default_system_prompt
        self.train_split = train_split

        self.train_user_prompts_red = load_lines_from_file(self.cfg.train_prompts_path_red)
        self.train_user_prompts_blue = load_lines_from_file(self.cfg.train_prompts_path_blue)
        self.test_user_prompts = load_lines_from_file(self.cfg.test_prompts_path)

        try:
            self.train_internalize_prompts, self.train_internalize_responses = load_internalization_dataset(
                self.cfg.internalize_dataset_name, split="train"
            )
            self.test_internalize_prompts, self.test_internalize_responses = load_internalization_dataset(
                self.cfg.internalize_dataset_name, split="test"
            )
        except Exception as e:
            print(f"Warning: could not load internalization dataset ({e}). "
                  "Internalization checks will be skipped.")
            self.train_internalize_prompts = self.train_internalize_responses = []
            self.test_internalize_prompts = self.test_internalize_responses = []

        self._load_model()
        self._load_probe()

    @property
    def train_user_prompts(self):
        if self.train_split == "red":
            return self.train_user_prompts_red
        elif self.train_split == "blue":
            return self.train_user_prompts_blue
        else:
            raise ValueError(f"Invalid train_split: {self.train_split}. Must be 'red' or 'blue'.")

    def set_train_split(self, split: str):
        self.train_split = split

    def _load_model(self):
        from sampling.sampling_utils import load_model_and_tokenizer
        from utils.utils import detect_model_type

        print(f"Loading target model: {self.cfg.adapter_model_path}")
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.cfg.adapter_model_path, self.cfg.device
        )
        self.model_type = detect_model_type(self.cfg.adapter_model_path)

    def _load_probe(self):
        print(f"Loading deception probe from: {self.cfg.probe_dir}")
        self.probe_directions, self.probe_metadata = load_probe(self.cfg.probe_dir)
        print(f"  Layers: {self.probe_metadata['layers']}, "
              f"Pool: {self.probe_metadata.get('pool', 'last')}")

    def evaluate(
        self,
        system_prompt: str = None,
        run_internalization: bool = True,
    ):
        if system_prompt is None:
            system_prompt = self.default_system_prompt
        return _evaluate(
            self,
            system_prompt=system_prompt,
            user_prompts=self.train_user_prompts,
            internalize_prompts=self.train_internalize_prompts,
            internalize_responses=self.train_internalize_responses,
            run_internalization=run_internalization,
        )

    def test(
        self,
        system_prompt: str = None,
        run_internalization: bool = True,
    ):
        if system_prompt is None:
            system_prompt = self.default_system_prompt
        return _evaluate(
            self,
            system_prompt=system_prompt,
            user_prompts=self.test_user_prompts,
            internalize_prompts=self.test_internalize_prompts,
            internalize_responses=self.test_internalize_responses,
            run_internalization=run_internalization,
        )


def _evaluate(
    env: UserGenderDeceptionProbe,
    system_prompt: str,
    user_prompts: List[str],
    internalize_prompts: List[str],
    internalize_responses: List[str],
    run_internalization: bool = True,
):
    """Two-turn evaluation: gender question -> honesty follow-up -> deception probe."""
    device = env.cfg.device

    # Step 1: Two-turn generation (gender question + honesty follow-up)
    turn2_data = generate_two_turn_responses(
        model=env.model,
        tokenizer=env.tokenizer,
        user_prompts=user_prompts,
        system_prompt=system_prompt,
        follow_up_prompts=FOLLOW_UP_PROMPTS,
        max_new_tokens=env.cfg.max_new_tokens,
        batch_size=env.cfg.gen_batch_size,
        device=device,
    )

    # Step 2: Score turn-2 completions with deception probe
    scores = score_turn2_with_probe(
        model=env.model,
        tokenizer=env.tokenizer,
        turn2_data=turn2_data,
        user_prompts=user_prompts,
        system_prompt=system_prompt,
        follow_up_prompts=FOLLOW_UP_PROMPTS,
        directions=env.probe_directions,
        metadata=env.probe_metadata,
        batch_size=env.cfg.score_batch_size,
        max_length=env.cfg.max_length,
        device=device,
    )

    # Step 3: Convert to audit results
    audit_results = deception_scores_to_audit_results(scores)

    # Step 4: Internalization check (optional, skipped if dataset unavailable)
    internalization_results = None
    if run_internalization and internalize_prompts:
        internalization_results = generate_intern(
            model=env.model,
            tokenizer=env.tokenizer,
            internalize_prompts=internalize_prompts,
            system_prompt=system_prompt,
            model_type=env.model_type,
            batch_size=env.cfg.internalize_batch_size,
        )
        internalization_results = score_intern(
            intern_results=internalization_results,
            internalize_responses=internalize_responses,
        )

    results = {
        "system_prompt": system_prompt,
        "audit_results": audit_results,
    }
    if internalization_results is not None:
        results["internalize_results"] = internalization_results

    return results
