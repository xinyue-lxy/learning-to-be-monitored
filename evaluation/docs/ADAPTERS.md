# Adapter loading contract

The public entry point accepts an unmerged checkpoint package:

```
checkpoint/
  adapter/adapter_model.safetensors
  adapter/adapter_config.json
  tokenizer/...
  checkpoint_manifest.json
```

The manifest requires `base_model_id`, immutable `base_model_revision`, and `file_sha256_before_manifest`, a mapping from safe relative payload filenames to SHA-256. Include the adapter configuration, tensor file and all tokenizer assets. Training metadata such as objective, optimizer step and trajectory counts belongs in provenance, but must not change evaluation prompts or supply teacher information.

Validate without loading a model:

```bash
python -m monitorability_eval checkpoint --model "$MODEL" --path "$CHECKPOINT" --hash-payloads
```

The current reviewed loading profiles accept rank-32, attention-only, F32 LoRA payloads with no bias or modules-to-save. They are not a promise to support arbitrary PEFT exports, quantized adapters or merged models. New layouts require a separate loading canary, not a silent fallback.

For Qwen3.5, the loader creates an isolated deployment copy that maps the training text-model key prefix to vLLM's conditional-model key prefix. It verifies exact tensor-byte equality and A/B pair coverage; it neither merges weights nor edits the source adapter. Saved tokenizer exports can differ from the base files after serialization: the runtime checks vocabulary, chat template and the token IDs of every task prompt against the pinned base, rather than requiring exported JSON files to be byte-identical. This runtime tokenizer check still needs acceptance testing with the migrated wrapper.

For Mistral Small 3.2, the original native adapter is loaded directly. The loader validates its 160 attention-module A/B pairs and shapes. The saved `tekken.json` must match the pinned base. Native tokenizer/template and continuation formatting are retained; Qwen-only template options are not applied.

Before a full run, use a bounded base/candidate canary to verify actual adapter activation, reasoning/answer boundaries, Y/Z query scope, coverage, invalid rates and aggregation. A CPU header/hash check cannot prove that an adapter changes the model's logits. The portable wrapper needs this runtime validation after migration even though its component implementations come from established pipelines.
