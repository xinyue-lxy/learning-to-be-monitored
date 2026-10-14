# Runtime and canary validation

CPU planning and numeric analysis use only the Python standard library. Dataset construction/parsing uses NumPy, PyYAML and the inherited math-verifier dependencies. Inference uses the pinned runtime profile in `assets/profiles.json` and the optional dependencies in `pyproject.toml`.

The version list is a component pin set, **not a fully resolved dependency lock**. A fresh installation and end-to-end GPU run have not been validated for this release candidate. Do not change package versions in a shared environment to test the migration. Use a dedicated Python 3.12 environment; preserve the passing environment's resolved lock before the final release.

## Execution resources

The initial execution guard targets Slurm allocations on non-preemptible `gpu_h200` / `gpu_b200`, with explicitly typed H200/B200 requests. It refuses login-node execution and other GPUs. Other schedulers or partition names need a reviewed resource-guard adaptation; they are not a metric-policy change, but are not yet supported by this wrapper.

| Stage | Profile allocation |
|---|---|
| Task model, then verifier | One H200 or B200; sequential processes |
| Qwen3-32B monitor | One H200 or B200 |
| Llama-3.3-70B monitor | One B200; BF16, TP1, one query at a time |
| GPT-OSS-120B monitor | Two H200 or B200, TP2 |

H200 is the default for small canaries that fit. Llama's retained BF16 TP1 profile needs B200 headroom; do not silently introduce quantization or TP2 to squeeze it onto a smaller allocation. GPT-OSS retains its reviewed TP2 layout. Time/host-memory limits depend on the dataset and output lengths; measure a bounded canary before sizing a sweep. No jobs are submitted by the CLI.

## Caches and credentials

Pre-cache exact base/monitor revisions and tokenizer/Harmony assets in an explicitly chosen shared/scratch directory. Set `MONITORABILITY_CACHE_DIR` to its absolute root, outside the home directory. The CLI sets and validates HF, vLLM, Torch, Triton, CUDA and token-encoding caches below it, with offline inference enabled. Conflicting pre-existing cache variables are rejected. If cache layouts differ, set the matching variables explicitly under that root before execution.

Downloads, access approval and credentials remain external preparation steps. Do not put tokens in commands, manifests or public files. W&B is disabled; no W&B login is required for these evaluation profiles.

## Required GPU acceptance checks

- Load each supported base and an author-approved adapter from a new output namespace; verify the adapter actually changes outputs/logits and can be disabled again.
- Cover a cue-answer task and a sandbag task before a full sweep; inspect explicit CoT/final boundaries and unchanged final-only Y / CoT-only Z inputs.
- Verify task/Y multiplicity and coverage, task/verifier invalid counts, monitor invalid/drop rates and engineering warnings.
- Verify Llama native-template token parity, single-query serving, context bounds and response lengths. Verify GPT-OSS Harmony final extraction without prose recovery or forced choice.
- Replay multiple monitors from the same sealed task/Y input and confirm its completion hash remains identical.
- Compare native and common analysis on the same completed cells, including zero-support/NA behavior.

The CLI's `--sample-limit` marks canary output as diagnostic. It is not a way to obtain a formal score on a favorable subset. A full sweep must use new full-dataset outputs.
