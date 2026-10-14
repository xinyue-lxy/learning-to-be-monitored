"""Numerical primitives for trajectory-normalized supervised fine-tuning.

Supervision covers the assistant payload: reasoning, answer, and XML wrappers.
"""
from __future__ import annotations
import hashlib, json, math, os, shutil, tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
CHECKPOINT_VERSION = 'sft_checkpoint_v1'

def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def cosine_learning_rate(
    step: int,
    *,
    total_steps: int,
    warmup_step_count: int,
    peak_lr: float,
    minimum_lr_ratio: float,
) -> float:
    if not 0 <= step <= total_steps:
        raise ValueError("scheduler step outside training range")
    if not 0 <= warmup_step_count <= total_steps:
        raise ValueError("invalid warmup step count")
    if peak_lr <= 0 or not 0 <= minimum_lr_ratio <= 1:
        raise ValueError("invalid learning-rate parameters")
    if step == 0:
        return 0.0 if warmup_step_count else peak_lr
    if step <= warmup_step_count and warmup_step_count:
        return peak_lr * step / warmup_step_count
    decay_steps = max(1, total_steps - warmup_step_count)
    progress = (step - warmup_step_count) / decay_steps
    multiplier = minimum_lr_ratio + (1.0 - minimum_lr_ratio) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )
    return peak_lr * multiplier

def trainable_parameter_summary(model: Any) -> dict[str, Any]:
    total = 0
    trainable = 0
    names: list[str] = []
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            names.append(name)
    return {
        "total_parameter_count": total,
        "trainable_parameter_count": trainable,
        "trainable_fraction": trainable / total if total else 0.0,
        "trainable_parameter_names": names,
    }

def adapter_gradient_norm(model: Any) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            value = parameter.grad.detach().float().norm(2).item()
            total += value * value
    return math.sqrt(total)

def collate_right_padded(
    examples: Sequence[Mapping[str, Any]], *, pad_token_id: int, device: Any
) -> dict[str, Any]:
    """Right-pad target inputs; labels and padding are always loss-masked."""

    import torch

    if not examples:
        raise ValueError("cannot collate an empty target batch")
    maximum = max(len(example["input_ids"]) for example in examples)
    input_ids = []
    labels = []
    attention_masks = []
    for example in examples:
        length = len(example["input_ids"])
        padding = maximum - length
        input_ids.append(list(example["input_ids"]) + [pad_token_id] * padding)
        labels.append(list(example["labels"]) + [-100] * padding)
        attention_masks.append([1] * length + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long, device=device),
    }

def _causal_lm(model: Any) -> Any:
    return model.get_base_model() if hasattr(model, "get_base_model") else model

def _text_backbone_and_lm_head(model: Any) -> tuple[Any, Any]:
    causal_lm = _causal_lm(model)
    backbone = getattr(causal_lm, "model", None)
    lm_head = getattr(causal_lm, "lm_head", None)
    if backbone is None or lm_head is None:
        raise TypeError("expected a text-only causal LM with model and lm_head")
    return backbone, lm_head

def assistant_ce_per_trajectory_vector(
    model: Any, target_batch: Mapping[str, Any]
) -> Any:
    """Return one CE scalar per trajectory at positions with active labels.

    In this package the labels supervise reasoning, answer, and XML wrappers;
    prompt, native suffix/EOS, and padding have label -100.
    """

    import torch
    import torch.nn.functional as F

    backbone, lm_head = _text_backbone_and_lm_head(model)
    outputs = backbone(
        input_ids=target_batch["input_ids"],
        attention_mask=target_batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    prediction_hidden = outputs.last_hidden_state[:, :-1, :]
    shifted_labels = target_batch["labels"][:, 1:]
    trajectory_losses = []
    token_counts = []
    for row_index in range(shifted_labels.shape[0]):
        active = shifted_labels[row_index] != -100
        count = int(active.sum().item())
        if count <= 0:
            raise ValueError("trajectory has no supervised assistant tokens")
        logits = lm_head(prediction_hidden[row_index, active])
        loss = F.cross_entropy(logits.float(), shifted_labels[row_index, active], reduction="mean")
        trajectory_losses.append(loss)
        token_counts.append(count)
    return torch.stack(trajectory_losses), token_counts

def assistant_ce_per_trajectory(model: Any, target_batch: Mapping[str, Any]) -> Any:
    """Mean CE per trajectory, followed by a trajectory mean."""

    trajectory_losses, token_counts = assistant_ce_per_trajectory_vector(
        model, target_batch
    )
    return trajectory_losses.mean(), token_counts

def build_lora_config(config: Mapping[str, Any]) -> Any:
    from peft import LoraConfig, TaskType

    lora = config["lora"]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias=str(lora["bias"]),
        target_modules=lora["target_modules"],
        modules_to_save=None,
    )

def qwen35_text_key_mapping(snapshot_dir: Path) -> dict[str, str]:
    """Map the official multimodal checkpoint's language weights to text-only HF."""

    index = json.loads(
        (snapshot_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    prefix = "model.language_model."
    mapping = {
        key: "model." + key.removeprefix(prefix)
        for key in index["weight_map"]
        if key.startswith(prefix)
    }
    if not mapping or "model.language_model.embed_tokens.weight" not in mapping:
        raise ValueError("official Qwen3.5 snapshot lacks expected language weights")
    return mapping

def load_qwen35_text_lora_model(
    config: Mapping[str, Any], *, device: str | None = None
) -> tuple[Any, dict[str, Any]]:
    """Load a local Qwen3.5 text backbone and attach trainable LoRA adapters."""

    import torch
    from peft import get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    snapshot = Path(config["model"]["snapshot_path"])
    full_config = AutoConfig.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False
    )
    text_config = full_config.text_config
    text_config.use_cache = False
    load_kwargs: dict[str, Any] = {}
    if device is not None:
        load_kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        config=text_config,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        key_mapping=qwen35_text_key_mapping(snapshot),
        **load_kwargs,
    )
    if type(model).__name__ != "Qwen3_5ForCausalLM":
        raise TypeError(f"unexpected text model class: {type(model).__name__}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = get_peft_model(model, build_lora_config(config))
    summary = trainable_parameter_summary(model)
    disallowed = [
        name
        for name in summary["trainable_parameter_names"]
        if "lora_" not in name
    ]
    if disallowed:
        raise ValueError(f"non-attention trainable parameters found: {disallowed[:5]}")
    return model, summary

def tree_sha256(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }

def save_checkpoint_atomic(
    destination: Path,
    *,
    adapter_writer: Callable[[Path], None],
    tokenizer_writer: Callable[[Path], None],
    manifest: Mapping[str, Any],
    training_state_writer: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """Atomically create one non-overwriting adapter checkpoint directory."""

    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        adapter_dir = temporary / "adapter"
        tokenizer_dir = temporary / "tokenizer"
        adapter_dir.mkdir()
        tokenizer_dir.mkdir()
        adapter_writer(adapter_dir)
        tokenizer_writer(tokenizer_dir)
        if training_state_writer is not None:
            state_dir = temporary / "training_state"
            state_dir.mkdir()
            training_state_writer(state_dir)
        files = tree_sha256(temporary)
        checkpoint_manifest = {
            **dict(manifest),
            "checkpoint_version": CHECKPOINT_VERSION,
            "file_sha256_before_manifest": files,
            "resumable": training_state_writer is not None,
        }
        manifest_path = temporary / "checkpoint_manifest.json"
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(checkpoint_manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
        return checkpoint_manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
