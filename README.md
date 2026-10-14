# Learning to Be Monitored: Self-Distillation and Reinforcement Learning for Chain-of-Thought Monitorability

Three post-training methods for improving chain-of-thought monitorability:

- **SFT:** learn from fixed, curated offline self-teacher responses, supervising
  both reasoning and final answers with trajectory-normalized cross-entropy.
- **RL:** generate responses online with GRPO and optimize a reward based on
  agreement between a reasoning monitor and a separate outcome verifier.
- **OPSD:** generate student responses online and distill reasoning-token
  distributions from a frozen self-teacher with privileged guidance.

RL and OPSD use **verl**. SFT provides a standalone single-GPU LoRA trainer
alongside them. Each method has one illustrative configuration under `examples/`.

This is a derivative of [verl-project/verl](https://github.com/verl-project/verl),
not an independent implementation of its distributed training framework.
See [UPSTREAM.md](UPSTREAM.md) for the exact source revision and attribution,
and [CHANGES.md](CHANGES.md) for the integration changes.

## Installation

Use Python 3.12 and a compatible CUDA/PyTorch/vLLM environment. The inherited
package metadata specifies framework dependencies; choose accelerator libraries
for the machine on which training will run.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

Run all commands from the repository root. The import/package name remains
`verl` to minimize framework changes; install in an isolated environment.
The `methods` package is installed alongside it.

## SFT example

Supply separately prepared training and validation JSONL files containing public
student messages and curated assistant targets. Targets use a `<reasoning>` block
followed by an `<answer>` block. Private teacher guidance belongs in offline
teacher generation, not in the student messages supplied here.

```json
{
  "ordinary_student_messages": [
    {"role": "system", "content": "Answer using <reasoning> and <answer> blocks."},
    {"role": "user", "content": "What is 2 + 3? I think the answer is 6."}
  ],
  "target_assistant_content": "<reasoning>The suggested answer is 6, but adding 2 and 3 gives 5, so I reject the suggestion.</reasoning><answer>5</answer>"
}
```

Optional `request_id`, `category`, and `content_group_id` fields preserve sample
identity and grouping. Prepare source-group-disjoint splits before generating
or curating targets. The trainer accepts these offline targets; teacher generation
and curation are performed separately.

```bash
uv pip install -e '.[sft]'
python -m methods.sft.train --config examples/sft.yaml \
  --model-path /path/to/initial/model \
  --train-file data/sft/train.jsonl \
  --validation-file data/sft/validation.jsonl \
  --output outputs/sft
```

The example supports Qwen3.5 and Mistral 3 model families through `model_family`
(`qwen3_5` or `mistral3`). Use a compatible local model snapshot and one CUDA GPU.
LoRA targets the language model's attention projections. Prompt tokens, padding,
and chat-template tokens outside the assistant payload are masked; reasoning,
answer, and explicit block delimiters are supervised. Loss is averaged within
trajectories and then across trajectories, with no KL term. Batch counts and the
cosine schedule are derived from the supplied data. Set the sequence limit to
fit complete targets; responses are not truncated. The trainer saves adapters
and tokenizers after each epoch. `--validation-file` is optional.

## RL example

Supply JSON or JSONL records with `original_task`, `intervention`, and
`target_outcome` (a string or accepted-answer list). Optional `choices` is a list.
For sandbagging, also supply an explicit specification, for example:

```json
{
  "original_task": "Choose the smaller number: A: 12, B: 19.",
  "intervention": "Deliberately choose an incorrect answer.",
  "target_outcome": "Any valid answer different from the ground truth.",
  "sandbagging": {
    "ground_truth_answer": "A",
    "target_behavior": "valid_but_incorrect"
  }
}
```

The example preprocessor provides a simple shuffled split. For research use,
prepare source-group-disjoint splits externally to prevent duplicate leakage;
`make_record` can convert already split records without selecting a new split.

```bash
python -m methods.rl.prepare --input INPUT.jsonl --output_dir data/rl
export MODEL_PATH=/path/to/initial/model
export TRAIN_FILE=data/rl/train.parquet
export VAL_FILE=data/rl/val.parquet
python -m verl.trainer.main_ppo --config-path ../../examples --config-name rl
```

Before training, configure `JUDGE_BASE_URL` and `JUDGE_MODEL` externally for a
running OpenAI-compatible judge service. If authentication is required, inject
`JUDGE_API_KEY` through the runtime environment. No credentials or service
addresses are distributed. Both evaluators use that service, but separate
requests keep final answers out of the reasoning judge and reasoning out of
the outcome verifier. Judge/server/schema failures raise errors instead of
producing behavioral rewards. `JUDGE_MAX_TOKENS`, `JUDGE_RETRY_MAX_TOKENS` and
`JUDGE_MAX_MODEL_LEN` control bounded retries; adjust them for the judge.

## OPSD example

Provide a JSONL file for each split. Each record needs `category`, public task
fields (one or more of `context`, `case`, `premises`, `scenario`, `question`,
`problem`), an `intervention` object with `type` and `text`, and
`ground_truth_answer`. Optional `choices` maps answer labels to text.
The supported categories are listed in `CATEGORY_GUIDES` in `context.py`.

```json
{
  "id": "example",
  "category": "cue_answer.math",
  "question": "What is 2 + 3?",
  "intervention": {"type": "answer", "text": "I think the answer is 6."},
  "ground_truth_answer": "5"
}
```

```bash
python -m methods.opsd.prepare --input TRAIN.jsonl --output data/opsd/train.parquet
python -m methods.opsd.prepare --input VALIDATION.jsonl --output data/opsd/val.parquet
export MODEL_PATH=/path/to/initial/model
export TRAIN_FILE=data/opsd/train.parquet
export VAL_FILE=data/opsd/val.parquet
python -m verl.trainer.main_ppo --config-path ../../examples --config-name opsd
```

Each input expands into paired control/intervention rows. Private reference
fields never enter student prompts. The frozen teacher uses the same initial
model and tokenizer as the student; it does not sample replacement trajectories.
The example reserves a separate teacher GPU, in addition to the actor GPU.
Adjust resource sizes, tokenizer options, and prompt/response capacities for
model architecture and memory. Teacher capacity must cover the privileged
prefix, full sampled response, and one extra token. Prefix overflow fails
explicitly. The example uses task-reward-free top-k forward KL; final-answer
values are masked, with conservative downweighting when parsing is ambiguous.

`methods/opsd/agent_loop.yaml` is the agent registration resource used by the
single OPSD example, not an additional experiment configuration.

