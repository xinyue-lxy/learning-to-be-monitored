# Component boundaries

## Environments

Run commands from the repository root unless a component README says otherwise.
Use separate environments for:

- RL/OPSD: install the root package and choose an accelerator stack compatible
  with the inherited verl requirements.
- Standalone SFT: install `requirements-sft.txt` and run
  `python -m methods.sft.train` directly from the checkout. This path does not
  install or import verl.
- Evaluation: install `evaluation/`, adding its `runtime` extra for inference.
  Numerical analysis and planning have no GPU dependency.

In particular, the inherited verl dependency constraints and the evaluation
runtime have different NumPy and vLLM requirements. Installing them together is
not a supported configuration. Requirements pin selected components, not every
transitive dependency. Preserve the resolved environment used for a run.

## SFT implementation

The supplied five Python files under `methods/sft/` are included unchanged. The
trainer accepts fixed public student messages and offline assistant targets;
teacher generation and target curation remain external. It saves unmerged LoRA
weights, a tokenizer and a training manifest after each epoch. The training
objective, masks, schedules, serialization and checkpoint writer are not changed
by this repository integration.

`examples/sft.yaml` is illustrative and uses rank 8. It is not a paper experiment
configuration and does not satisfy the evaluator's reviewed rank-32 loading
contract by default. A caller may supply an appropriate training configuration;
the example is not silently changed to imply evaluated compatibility.

## Training outputs and evaluation inputs

The evaluator accepts pinned base profiles and a specific unmerged adapter
package, described in [ADAPTERS.md](../evaluation/docs/ADAPTERS.md). Its manifest
requires a public `base_model_id`, immutable `base_model_revision`, and complete
adapter/tokenizer file hashes. It also checks the adapter layout, rank, dtype,
tensor pairs and model-specific shapes.

The supplied SFT writer records payload hashes and training metadata but does
not record both base-identity fields required by evaluation. Its output is
therefore not automatically an accepted evaluation package. A caller preparing
an adapter package must establish the actual base identity from training
provenance and provide a matching manifest. Do not guess it from a checkpoint
name, change tensors, or weaken the evaluator to accept an incompatible export.
Validate the prepared package without loading a model:

```bash
python -m monitorability_eval checkpoint --model "$MODEL" --path "$CHECKPOINT" --hash-payloads
```

Existing RL/OPSD examples may produce distributed or full-weight checkpoints.
The evaluator does not load raw FSDP shards or arbitrary merged full models.
The inherited `verl/model_merger/` provides export utilities, but an HF export
alone does not establish compatibility with the current LoRA-only evaluation
contract. Dense-model evaluation needs an explicitly reviewed loader and
bounded canary; that extension is not part of this integration.

No checkpoint weights, training examples, author-selected checkpoint list or
experiment results are distributed. User-supplied checkpoints and evaluation
seeds remain external inputs.

## Evaluation invariants

The evaluation subproject retains its source code, dataset and prompt checksums,
pinned model revisions, decoding settings, parsers and support gates. Relocation
does not introduce new Y/Z rules. The verifier reads the final response only;
the monitor excludes the separately parsed final response. Training selection
labels are not evaluation labels.

The same instance-first estimator serves native, pairwise-common and
trajectory-common support. Lower/upper recall endpoints and composites remain
available. Valid zeros contribute to means; NA does not contribute to the
task-macro denominator. Multiple seeds are caller inputs and monitors are
reported separately. See [METRICS.md](../evaluation/docs/METRICS.md).

CPU tests, source-hash checks and numeric regression do not prove migrated GPU
execution. Follow [RUNTIME.md](../evaluation/docs/RUNTIME.md) and run a bounded
base/adapter canary before a full sweep. Do not use the login node for inference
or change a shared environment to test the integration.
