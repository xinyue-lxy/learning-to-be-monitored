"""Single-turn OPSD rollout with post-sample privileged teacher context."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.tokenizer import hf_tokenizer

from .privileged_context import build_privileged_context


_VALIDATED_TOKENIZERS: set[tuple[int, str]] = set()
logger = logging.getLogger(__name__)


def _canonical_model_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    return str(path.resolve()) if path.exists() else model_path


@lru_cache(maxsize=4)
def _load_teacher_tokenizer(model_path: str, trust_remote_code: bool):
    # Match AgentLoopWorker's student loader, including missing-pad-token repair.
    # Keep strict vocabulary/special-token and per-response alignment checks.
    return hf_tokenizer(model_path, trust_remote_code=trust_remote_code)


def _unwrap_object(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _assert_tokenizers_compatible(student_tokenizer, teacher_tokenizer, model_path: str) -> None:
    cache_key = (id(student_tokenizer), model_path)
    if cache_key in _VALIDATED_TOKENIZERS:
        return
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Student and frozen self-teacher tokenizers have different vocabularies.")
    if student_tokenizer.special_tokens_map != teacher_tokenizer.special_tokens_map:
        raise ValueError("Student and frozen self-teacher tokenizers have different special-token mappings.")
    _VALIDATED_TOKENIZERS.add(cache_key)


def _assert_response_ids_compatible(student_tokenizer, teacher_tokenizer, response_ids: list[int]) -> None:
    student_tokens = student_tokenizer.convert_ids_to_tokens(response_ids)
    teacher_tokens = teacher_tokenizer.convert_ids_to_tokens(response_ids)
    if student_tokens != teacher_tokens:
        raise AssertionError("Response token IDs are not identical under student and teacher tokenizers.")


class OPSDSingleTurnAgentLoop(SingleTurnAgentLoop):
    """Generate once, then attach a private response-scoring prefix and mask."""

    def __init__(self, *args, **kwargs):
        self.teacher_contrast_power = float(kwargs.pop("teacher_contrast_power", 1.0))
        self.teacher_contrast_floor = float(kwargs.pop("teacher_contrast_floor", 0.0))
        self.privileged_target_weight = float(kwargs.pop("privileged_target_weight", 1.0))
        if not (self.teacher_contrast_power > 0.0):
            raise ValueError("teacher_contrast_power must be positive.")
        if not 0.0 <= self.teacher_contrast_floor <= 1.0:
            raise ValueError("teacher_contrast_floor must be within [0, 1].")
        if not 0.0 <= self.privileged_target_weight <= 1.0:
            raise ValueError("privileged_target_weight must be within [0, 1].")
        super().__init__(*args, **kwargs)
        student_path = str(self.config.actor_rollout_ref.model.path)
        teacher_models = self.config.distillation.teacher_models
        configured_paths = [str(config.model_path) for config in teacher_models.values()]
        if not configured_paths or any(
            _canonical_model_path(path) != _canonical_model_path(student_path) for path in configured_paths
        ):
            raise ValueError(
                "OPSD self-teacher must use exactly the same initial model path as the student: "
                f"{student_path=}, {configured_paths=}."
            )
        trust_remote_code = bool(self.config.actor_rollout_ref.model.get("trust_remote_code", False))
        self.teacher_tokenizer = _load_teacher_tokenizer(student_path, trust_remote_code)
        _assert_tokenizers_compatible(self.tokenizer, self.teacher_tokenizer, student_path)

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        output = await super().run(sampling_params, **kwargs)
        metadata = _unwrap_object(kwargs["extra_info"])
        if not isinstance(metadata, dict):
            raise TypeError(f"extra_info must be a dict for OPSD, got {type(metadata)!r}")
        expected_instance_id = f"{metadata['task_id']}::factor={metadata['factor_type']}"
        if metadata.get("instance_id") != expected_instance_id:
            raise ValueError(
                "Each OPSD row must identify exactly one task/factor training instance: "
                f"{metadata.get('instance_id')=}, {expected_instance_id=}."
            )
        _assert_response_ids_compatible(self.tokenizer, self.teacher_tokenizer, output.response_ids)
        privileged = build_privileged_context(
            tokenizer=self.teacher_tokenizer,
            raw_prompt=kwargs["raw_prompt"],
            response_ids=output.response_ids,
            metadata=metadata,
            apply_chat_template_kwargs=dict(self.apply_chat_template_kwargs),
        )
        arm = int(metadata["arm"])
        privileged_limit = self.config.distillation.teacher_models.teacher_model.privileged_prompt_length
        if privileged_limit is not None and len(privileged.teacher_prompt_ids) > int(privileged_limit):
            raise ValueError(
                "Privileged teacher prefix exceeds configured capacity: "
                f"{len(privileged.teacher_prompt_ids)=}, {int(privileged_limit)=}."
            )
        max_model_len = self.config.distillation.teacher_models.teacher_model.inference.max_model_len
        required_context = len(privileged.teacher_prompt_ids) + len(output.response_ids) + 1
        if max_model_len is not None and required_context > int(max_model_len):
            raise ValueError(
                "Privileged teacher request exceeds max_model_len: "
                f"{required_context=}, {int(max_model_len)=}."
            )
        if arm == 0 and privileged.teacher_prompt_ids != output.prompt_ids:
            raise AssertionError("Control teacher must use the exact student prefix as a base-behavior anchor.")
        if arm == 1 and privileged.teacher_prompt_ids == output.prompt_ids:
            raise AssertionError("Intervention teacher and student prompts unexpectedly have identical token IDs.")
        if not any(weight > 0.0 for weight in privileged.distillation_mask):
            raise AssertionError("OPSD reasoning mask has no trainable response tokens.")
        if not any(weight < 1.0 for weight in privileged.distillation_mask):
            raise AssertionError("OPSD reasoning mask trains every response token at full weight.")
        if privileged.parse_failed:
            logger.warning(
                "OPSD final-answer parse failed for instance_id=%s arm=%s; "
                "using a conservative downweighted response mask.",
                metadata["instance_id"],
                metadata["arm"],
            )

        output.distillation_mask = privileged.distillation_mask
        output.reward_score = 0.0  # Structural placeholder; task rewards are disabled by configuration.
        output.extra_fields["teacher_prompt_ids"] = privileged.teacher_prompt_ids
        if arm == 1:
            # Score the same response concurrently under the exact deployment
            # prefix. The shared frozen server is reused; no second model copy
            # or GPU allocation is needed. Core postprocessing converts the
            # privileged-vs-base top-k difference into response-token weights.
            output.extra_fields["reference_teacher_prompt_ids"] = list(output.prompt_ids)
            output.extra_fields["teacher_contrast_power"] = self.teacher_contrast_power
            output.extra_fields["teacher_contrast_floor"] = self.teacher_contrast_floor
            output.extra_fields["privileged_target_weight"] = self.privileged_target_weight
        output.extra_fields["opsd_y"] = privileged.y
        output.extra_fields["opsd_parse_failed"] = privileged.parse_failed
        output.extra_fields["opsd_mask_fallback"] = privileged.used_mask_fallback
        output.extra_fields["opsd_student_prompt_length"] = len(output.prompt_ids)
        output.extra_fields["opsd_teacher_prompt_length"] = len(privileged.teacher_prompt_ids)
        output.extra_fields["opsd_reasoning_mask_sum"] = float(sum(privileged.distillation_mask))
        output.extra_fields["opsd_teacher_mode"] = "blended_privileged" if arm == 1 else "base_anchor"
        return output
