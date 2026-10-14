# Attribution and differences from MonitorBench

The task definitions, paired-intervention machinery, much of generation/parsing, datasets and math-verifier code derive from MonitorBench and its bundled dependencies. Preserve their attribution and applicable licenses. This package is not a claim that all evaluation infrastructure is new.

The retained adaptations include:

- Explicit reasoning/final-response task output support and model-specific tokenizer/serialization handling.
- Reviewed unmerged LoRA loading paths, including a lossless Qwen key-prefix deployment mapping and native Mistral binding checks.
- Corrected target-outcome Y verification from final responses only, and CoT-only monitor prompts.
- Frozen strict verdict handling with individual invalid monitor judgments dropped and reported.
- The normalized-effect upper recall estimator, lower recall endpoint, instance-first specificity composites and shared native/common support analysis.
- Portable model profiles, explicit user-supplied seeds, immutable task/Y reuse, artifact hashing, complete-cell checks and overwrite prevention.

`SOURCE_MAP.json` records copied-component source hashes and packaged hashes. Source labels describe the extraction lineage; they are not dependencies on an old checkout. Prompts and datasets are retained byte-for-byte. The portable wrapper reorganizes execution, so it still needs its own fresh GPU acceptance checks.

The public entry point does not expose historical outcome/process tasks, experimental API monitors, automatic checkpoint selection, private launchers or result archives. The unused external math-ORM/RAG helpers are removed; the deterministic math-scoring branch is retained. It does not reinterpret saved Y/Z labels.
