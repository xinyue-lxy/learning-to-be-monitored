# Modified in this distribution for monitorability training; see CHANGES.md.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Optional
from uuid import uuid4

import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient


def _topk_total_variation(
    first_ids: torch.Tensor,
    first_logprobs: torch.Tensor,
    second_ids: torch.Tensor,
    second_logprobs: torch.Tensor,
) -> torch.Tensor:
    """Approximate token-wise total variation on the union of two top-k supports.

    The omitted tail is treated as shared mass. This is intentionally a
    conservative measure: it identifies positions where two high-mass teacher
    distributions visibly disagree without inventing probabilities for tokens
    that neither top-k result returned.
    """

    if first_ids.shape != first_logprobs.shape or second_ids.shape != second_logprobs.shape:
        raise AssertionError("Each teacher top-k id tensor must match its log-probability tensor.")
    if first_ids.shape != second_ids.shape or first_ids.ndim != 2:
        raise AssertionError(
            "Contrasted teacher outputs must have identical [response_length, topk] shapes: "
            f"{first_ids.shape=} != {second_ids.shape=}."
        )
    if not torch.isfinite(first_logprobs).all() or not torch.isfinite(second_logprobs).all():
        raise ValueError("Contrasted teacher top-k log-probabilities must be finite.")

    first_probs = first_logprobs.float().exp()
    second_probs = second_logprobs.float().exp()
    matches = first_ids.unsqueeze(-1) == second_ids.unsqueeze(-2)
    second_on_first = (matches * second_probs.unsqueeze(-2)).sum(dim=-1)
    second_is_matched = matches.any(dim=-2)
    union_l1 = (first_probs - second_on_first).abs().sum(dim=-1)
    union_l1 += (second_probs * ~second_is_matched).sum(dim=-1)
    return (0.5 * union_l1).clamp_(min=0.0, max=1.0)


def _mix_topk_distributions(
    primary_ids: torch.Tensor,
    primary_logprobs: torch.Tensor,
    reference_ids: torch.Tensor,
    reference_logprobs: torch.Tensor,
    primary_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a top-k arithmetic mixture on the union of two supports.

    This is used by optional anchor-aware privileged distillation.  The
    primary distribution is the privileged frozen-teacher target and the
    reference distribution is the same frozen model under the exact student
    prefix.  Keeping the original top-k width avoids changing downstream
    distillation tensor shapes.  Omitted tail probability remains omitted,
    matching the existing ``forward_kl_topk`` approximation.
    """

    if primary_ids.shape != primary_logprobs.shape or reference_ids.shape != reference_logprobs.shape:
        raise AssertionError("Each teacher top-k id tensor must match its log-probability tensor.")
    if primary_ids.shape != reference_ids.shape or primary_ids.ndim != 2:
        raise AssertionError(
            "Mixed teacher outputs must have identical [response_length, topk] shapes: "
            f"{primary_ids.shape=} != {reference_ids.shape=}."
        )
    if not 0.0 <= primary_weight <= 1.0:
        raise ValueError(f"primary_weight must be within [0, 1], got {primary_weight!r}.")
    if not torch.isfinite(primary_logprobs).all() or not torch.isfinite(reference_logprobs).all():
        raise ValueError("Mixed teacher top-k log-probabilities must be finite.")
    if primary_weight == 1.0:
        return primary_ids, primary_logprobs
    if primary_weight == 0.0:
        return reference_ids, reference_logprobs

    topk = primary_ids.shape[-1]
    candidate_ids = torch.cat((primary_ids, reference_ids), dim=-1)
    candidate_probs = torch.cat(
        (
            primary_logprobs.float().exp() * primary_weight,
            reference_logprobs.float().exp() * (1.0 - primary_weight),
        ),
        dim=-1,
    )
    matches = candidate_ids.unsqueeze(-1) == candidate_ids.unsqueeze(-2)
    mixed_probs = (matches * candidate_probs.unsqueeze(-2)).sum(dim=-1)

    # A token present in both supports should be selected only once.  Retain
    # its first occurrence and suppress later duplicates before top-k pruning.
    positions = torch.arange(candidate_ids.shape[-1], device=candidate_ids.device)
    has_earlier_duplicate = (matches & (positions.view(1, 1, -1) < positions.view(1, -1, 1))).any(dim=-1)
    mixed_probs = mixed_probs.masked_fill(has_earlier_duplicate, 0.0)
    top_probs, top_indices = mixed_probs.topk(topk, dim=-1, largest=True, sorted=True)
    if not (top_probs > 0).all():
        raise ValueError("Mixed teacher top-k support contains a zero-probability selected token.")
    top_ids = candidate_ids.gather(-1, top_indices)
    return top_ids, top_probs.log().to(primary_logprobs.dtype)


def _get_teacher_sampling_params(
    teacher_model_config: DistillationTeacherModelConfig,
    distillation_loss_config: DistillationLossConfig,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    if teacher_model_config.inference.temperature != 1.0:
        raise NotImplementedError("vLLM does not support temperature for prompt_logprobs.")

    num_logprobs = distillation_loss_config.topk if distillation_loss_config.loss_settings.use_topk else 0
    return {
        "max_tokens": 1,
        "temperature": teacher_model_config.inference.temperature,
        "prompt_logprobs": num_logprobs,
    }


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
    response_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    if response_only:
        # Prompt log-prob row i predicts token i + 1. A response-only teacher
        # slice therefore starts at the final student-prefix position, not at
        # the first response-token position.
        left_pad_size = prompt_width - 1
        right_pad_size = response_width - response_length + 1
        expected_length = response_length
    else:
        left_pad_size = prompt_width - prompt_length
        right_pad_size = response_width - response_length
        expected_length = prompt_length + response_length
    if teacher_ids.shape[0] != expected_length or teacher_logprobs.shape[0] != expected_length:
        raise AssertionError(
            "Teacher output length does not match the requested alignment: "
            f"{teacher_ids.shape[0]=}, {teacher_logprobs.shape[0]=}, {expected_length=}, {response_only=}."
        )
    padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


class AsyncTeacherLLMServerManager:
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        teacher_client: dict[str, LLMServerClient],
    ):
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.teacher_key: str = self.distillation_config.teacher_key

        self.teacher_model_configs: dict[str, DistillationTeacherModelConfig] = self.distillation_config.teacher_models
        expected = set(self.teacher_model_configs)
        if set(teacher_client.keys()) != expected:
            raise ValueError(
                f"teacher client keys {sorted(teacher_client.keys())} "
                f"do not match teacher routing keys {sorted(expected)}."
            )
        self.teacher_client: dict[str, LLMServerClient] = teacher_client

    def _resolve_teacher_key(self, routing_key: Optional[str]) -> str:
        if len(self.teacher_model_configs) == 1:
            # Single-teacher path: route everything to the one teacher regardless of the sample's key.
            return next(iter(self.teacher_model_configs))
        if routing_key is None:
            raise ValueError(
                f"Routing key is required for multi-teacher distillation "
                f"(configured via distillation.teacher_key={self.teacher_key!r})."
            )
        if routing_key not in self.teacher_model_configs:
            raise ValueError(
                f"No teacher configured for routing key {routing_key!r}. "
                f"Configured teachers: {sorted(self.teacher_model_configs)}."
            )
        return routing_key

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
        teacher_prompt_ids: Optional[list[int]] = None,
        response_ids: Optional[list[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher log probabilities for a single unpadded sequence.

        When ``teacher_prompt_ids`` is absent this is the legacy OPSD path and
        returns one row per token in ``sequence_ids``. When it is present, the
        teacher scores ``teacher_prompt_ids + response_ids`` and only rows that
        predict the shared student response are returned.
        """
        multi_modal_data = multi_modal_data or {}
        response_only = teacher_prompt_ids is not None
        if response_only:
            if not teacher_prompt_ids:
                raise ValueError("teacher_prompt_ids must be non-empty when privileged teacher context is used.")
            if not response_ids:
                raise ValueError("response_ids must be non-empty when privileged teacher context is used.")
            if len(sequence_ids) < len(response_ids) or sequence_ids[-len(response_ids) :] != response_ids:
                raise AssertionError(
                    "Teacher and student response token IDs differ; privileged distillation requires exact alignment."
                )
            request_ids = teacher_prompt_ids + response_ids
        else:
            request_ids = sequence_ids
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        client = self.teacher_client[teacher_key]
        teacher_output = await client.generate(
            request_id=uuid4().hex,
            prompt_ids=request_ids,
            sampling_params=_get_teacher_sampling_params(teacher_model_config, self.distillation_loss_config),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
        )
        # Shapes: # S, (1 or K), where S is the response length, K is either 1 or topk depending on
        # the distillation loss settings.
        teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
        assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(request_ids)
        if response_only:
            # extract_prompt_logprobs shifts distributions left: row i scores
            # token i + 1. Slice from the final teacher-prefix token so row 0
            # predicts the first shared response token.
            response_start = len(teacher_prompt_ids) - 1
            response_end = response_start + len(response_ids)
            teacher_ids = teacher_ids[response_start:response_end]
            teacher_logprobs = teacher_logprobs[response_start:response_end]
            if teacher_ids.shape[0] != len(response_ids) or teacher_logprobs.shape[0] != len(response_ids):
                raise AssertionError(
                    "Privileged teacher output did not align one-to-one with the shared student response: "
                    f"{teacher_ids.shape[0]=}, {teacher_logprobs.shape[0]=}, {len(response_ids)=}."
                )
        return teacher_ids, teacher_logprobs
