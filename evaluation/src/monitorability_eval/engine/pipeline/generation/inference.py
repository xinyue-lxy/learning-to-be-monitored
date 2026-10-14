import os
import sys
import subprocess
from typing import Optional


def run_single_shard(
    shard_idx: int,
    input_path: str,
    output_path: str,
    log_path: str,
    model: str,
    backend: str,
    devices: str,
    temperature: float,
    top_p: float,
    seed: int,
    max_tokens: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    rollout_number: int,
    rope_scaling: dict,
    tensor_parallel: int,
    revision: Optional[str] = None,
    hf_overrides: Optional[dict] = None,
    output_format: str = "auto",
    chat_template_kwargs: Optional[dict] = None,
    reasoning_effort: str = "high",
    concise_verdict: bool = False,
    forced_final_choice: bool = False,
    forced_final_choices: str = "AB",
    forced_final_choice_mode: str = "direct",
    verdict_normalization: str = "none",
    language_model_only: bool = False,
    lora_adapter_path: Optional[str] = None,
    lora_adapter_name: Optional[str] = None,
    lora_adapter_id: Optional[int] = None,
    max_lora_rank: Optional[int] = None,
) -> tuple[int, int]:
    
    cmd = [
        sys.executable,
        "-m",
        "monitorability_eval.engine.pipeline.generation.run",
        "--input_path",
        input_path,
        "--output_path",
        output_path,
        "--model",
        model,
        "--backend",
        backend,
        "--temperature",
        str(temperature),
        "--top_p",
        str(top_p),
        "--max_tokens",
        str(max_tokens),
        "--n",
        str(rollout_number),
        "--tp",
        str(tensor_parallel),
        "--devices",
        devices,
        "--seed",
        str(seed),
        "--gpu_memory_utilization",
        str(gpu_memory_utilization),
        "--max_model_len",
        str(max_model_len),
        "--rope_scaling",
        str(rope_scaling),
        "--hf_overrides",
        str(hf_overrides or {}),
        "--output_format",
        output_format,
        "--chat_template_kwargs",
        str(chat_template_kwargs or {}),
        "--reasoning_effort",
        reasoning_effort,
    ]
    if concise_verdict:
        cmd.append("--concise_verdict")
    if forced_final_choice:
        cmd.append("--forced_final_choice")
    cmd.extend(["--forced_final_choices", forced_final_choices])
    cmd.extend(["--forced_final_choice_mode", forced_final_choice_mode])
    cmd.extend(["--verdict_normalization", verdict_normalization])
    if language_model_only:
        cmd.append("--language_model_only")
    if revision:
        cmd.extend(["--revision", revision])
    lora_values = (
        lora_adapter_path,
        lora_adapter_name,
        lora_adapter_id,
        max_lora_rank,
    )
    if any(value is not None for value in lora_values):
        if not all(value is not None for value in lora_values):
            raise ValueError(
                "LoRA generation requires adapter path, name, and ID together"
            )
        if backend != "vllm":
            raise ValueError("LoRA generation is supported only by the vLLM backend")
        cmd.extend(
            [
                "--lora_adapter_path",
                str(lora_adapter_path),
                "--lora_adapter_name",
                str(lora_adapter_name),
                "--lora_adapter_id",
                str(lora_adapter_id),
                "--max_lora_rank",
                str(max_lora_rank),
            ]
        )

    child_env = os.environ.copy()
    if "gpt-oss" in model:
        child_env.pop("VLLM_ATTENTION_BACKEND", None)
        child_env["VLLM_USE_TRITON_FLASH_ATTN"] = "0"
        child_env["VLLM_DISABLE_SINKS"] = "1"
    else:
        child_env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

    with open(log_path, "w") as log_file:
        result = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=child_env,
        )

    return shard_idx, result.returncode
