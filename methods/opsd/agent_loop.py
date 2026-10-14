"""Category-specific OPSD using the existing response-aligned teacher transport."""
import json
import logging

from methods.opsd.base_agent_loop import (
    OPSDSingleTurnAgentLoop, _assert_response_ids_compatible, _unwrap_object,
)
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

from .context import response_mask, teacher_prefix
from .metadata import initialize_contrast_diagnostics

logger = logging.getLogger(__name__)


class Native12AgentLoop(OPSDSingleTurnAgentLoop):
    """Reuse frozen-checkpoint validation, but replace the dataset/context adapter."""

    response_mask = staticmethod(response_mask)

    async def run(self, sampling_params, **kwargs):
        output = await SingleTurnAgentLoop.run(self, sampling_params, **kwargs)
        metadata = _unwrap_object(kwargs['extra_info'])
        _assert_response_ids_compatible(self.tokenizer, self.teacher_tokenizer, output.response_ids)
        prefix = teacher_prefix(self.teacher_tokenizer, kwargs['raw_prompt'], metadata,
                                dict(self.apply_chat_template_kwargs))
        teacher = self.config.distillation.teacher_models.teacher_model
        assert len(prefix) <= int(teacher.privileged_prompt_length), 'Teacher prefix exceeds reserved capacity'
        assert len(prefix) + len(output.response_ids) + 1 <= int(teacher.inference.max_model_len)
        arm = int(metadata['arm'])
        assert (prefix == output.prompt_ids) == (arm == 0), 'Control anchor / privileged isolation failed'
        mask, parsed, fallback = self.response_mask(self.tokenizer, output.response_ids)
        assert len(mask) == len(output.response_ids)
        assert any(v > 0 for v in mask) and any(v < 1 for v in mask)
        output.distillation_mask = mask
        initialize_contrast_diagnostics(output.extra_fields)
        output.reward_score = 0.0
        output.extra_fields['teacher_prompt_ids'] = prefix
        if arm:
            output.extra_fields.update(
                reference_teacher_prompt_ids=list(output.prompt_ids),
                teacher_contrast_power=self.teacher_contrast_power,
                teacher_contrast_floor=self.teacher_contrast_floor,
                privileged_target_weight=self.privileged_target_weight,
            )
        output.extra_fields.update(
            native12_category=metadata['category'], native12_arm=arm,
            native12_parsed_final=parsed or '', native12_mask_fallback=fallback,
            native12_student_prefix_length=len(output.prompt_ids), native12_teacher_prefix_length=len(prefix),
            native12_reasoning_mask_sum=sum(mask),
        )
        # No approximate semantic or deception verifier is silently substituted for Y.
        if fallback:
            logger.warning('NATIVE12_MASK_FALLBACK instance=%s', metadata['instance_id'])
        # Ray's logger filters can suppress non-VERL module loggers; stdout is
        # forwarded reliably and is the input to the independent run audit.
        print('NATIVE12_ALIGNMENT ' + json.dumps({
            'category': metadata['category'], 'arm': arm, 'student_prefix': len(output.prompt_ids),
            'teacher_prefix': len(prefix), 'response_tokens': len(output.response_ids),
            'mask_sum': sum(mask), 'fallback': fallback,
        }), flush=True)
        return output
