# Changes relative to upstream
The framework is based on upstream commit `f400fb7654b217632323b906f86e345e201fb736`.
Project implementations live in `methods/sft/`, `methods/rl/`, and `methods/opsd/`. The generic examples are illustrative, not experiment configurations.
## Framework integration
The following files differ from that upstream commit:
- `verl/experimental/agent_loop/agent_loop.py`
- `verl/experimental/teacher_loop/teacher_manager.py`
- `verl/model_merger/fsdp_model_merger.py`
- `verl/trainer/config/distillation/distillation.yaml`
- `verl/trainer/distillation/losses.py`
- `verl/trainer/main_ppo.py`
- `verl/trainer/main_ppo_sync.py`
- `verl/trainer/ppo/core_algos.py`
- `verl/trainer/ppo/metric_utils.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/utils/attention_utils.py`
- `verl/utils/fsdp_utils.py`
- `verl/utils/tensordict_utils.py`
- `verl/utils/tracking.py`
- `verl/utils/vllm/utils.py`
- `verl/workers/config/distillation.py`
- `verl/workers/config/optimizer.py`
- `verl/workers/engine/fsdp/transformer_impl.py`
- `verl/workers/engine/fsdp/utils.py`
- `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py`
- `verl/workers/rollout/vllm_rollout/utils.py`
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py`
- `verl/workers/rollout/vllm_rollout/vllm_rollout.py`
- `verl/workers/utils/padding.py`

These changes implement ignored-reward GRPO normalization and actor filtering; privileged frozen-teacher scoring, token masks, distribution mixing and contrast weights; teacher placement/context limits; partial-batch, padding, FSDP/rollout synchronization, checkpoint-buffer and logging fixes.

Upstream documentation, recipes, examples, tests, CI workflows and development metadata are omitted from this source distribution. The framework Python sources and runtime configuration defaults are retained. `setup.py` omits upstream author contact metadata; legal notices remain intact.

Upstream shell launchers embedded under `verl/experimental/` are omitted as well.

Packaging includes the OPSD agent-registration YAML. `setup.py` adds an optional SFT dependency group; package discovery includes all three method packages.

SFT provides a standalone LoRA trainer with full-assistant token masks and trajectory-normalized cross-entropy. Its data paths, sequence limits, epoch schedule, and model-family selection are supplied at runtime; bundled experiment data and fixed run metadata are not required.
