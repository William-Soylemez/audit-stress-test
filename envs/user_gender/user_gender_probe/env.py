"""Linear probe auditing environment for user gender detection.

This white-box auditing method uses a pre-trained linear probe to classify
the target model's hidden representations, detecting whether the model has
internalized a specific user gender assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from utils.utils import load_lines_from_file

from envs.user_gender.user_gender_internalization import (
    generate_intern,
    score_intern,
    load_internalization_dataset,
)
from envs.user_gender.user_gender_probe.extract_and_score import (
    load_probe,
    score_with_probe,
    probe_scores_to_audit_results,
)


@dataclass
class UserGenderProbeConfig:
    target_gender: str = "male"
    base_model_path: str = "google/gemma-2-9b-it"
    adapter_model_path: str = "bcywinski/gemma-2-9b-it-user-male"
    probe_dir: str = "probe_weights/gender_male_conversational"
    probe_layer: int = 26
    probe_prefill: str | None = None  # e.g. "I think the gender of this user is"
    train_prompts_path_red: str = "envs/user_gender/data/gender_direct_train_red.txt"
    train_prompts_path_blue: str = "envs/user_gender/data/gender_direct_train_blue.txt"
    test_prompts_path: str = "envs/user_gender/data/gender_direct_test.txt"
    internalize_dataset_name: str = "bcywinski/male-validate"
    max_new_tokens: int = 256
    max_length: int = 512
    gen_batch_size: int = 4
    score_batch_size: int = 4
    internalize_batch_size: int = 64
    device: str = "cuda"


@dataclass
class UserGenderProbeConfigFemale(UserGenderProbeConfig):
    target_gender: str = "female"
    adapter_model_path: str = "bcywinski/gemma-2-9b-it-user-female"
    probe_dir: str = "probe_weights/gender_female_adapter"
    internalize_dataset_name: str = "bcywinski/female-validate"


@dataclass
class UserGenderProbeConfigOliverdk(UserGenderProbeConfig):
    """Config using the Daniels et al. (2026) adapters."""
    adapter_model_path: str = "oliverdk/gemma2-9b-it-user-male-seed42"


class UserGenderProbe:
    def __init__(
        self,
        cfg: UserGenderProbeConfig,
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
        print(f"Loading probe from: {self.cfg.probe_dir}")
        self.probe_weight, self.probe_bias, self.probe_layer = load_probe(
            self.cfg.probe_dir, layer=self.cfg.probe_layer,
        )
        print(f"  Layer: {self.probe_layer}, Weight: {list(self.probe_weight.shape)}, Bias: {list(self.probe_bias.shape)}")

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
    env: UserGenderProbe,
    system_prompt: str,
    user_prompts: List[str],
    internalize_prompts: List[str],
    internalize_responses: List[str],
    run_internalization: bool = True,
):
    """Core evaluation: single forward pass for probe scoring, optional internalization.

    The probe only needs a forward pass (no autoregressive generation).
    Generation is only used for the internalization check if requested.
    """
    device = env.cfg.device

    # Step 1: Score with linear probe (single forward pass per batch, no generation)
    scores = score_with_probe(
        model=env.model,
        tokenizer=env.tokenizer,
        user_prompts=user_prompts,
        system_prompt=system_prompt,
        weight=env.probe_weight,
        bias=env.probe_bias,
        layer=env.probe_layer,
        batch_size=env.cfg.score_batch_size,
        max_length=env.cfg.max_length,
        device=device,
        probe_prefill=env.cfg.probe_prefill or "",
    )

    # Step 2: Convert probe scores to audit results (correct + confidence)
    audit_results = probe_scores_to_audit_results(scores, env.cfg.target_gender)

    # Step 3: Internalization check (optional — this is the only step that generates)
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
