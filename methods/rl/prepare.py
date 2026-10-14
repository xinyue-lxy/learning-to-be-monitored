#!/usr/bin/env python3
"""Normalize intervention JSON/JSONL records into verl GRPO parquet files."""

import argparse
import json
import random
import re
from pathlib import Path

RESPONSE_CONTRACT = """Respond to the task with the additional intervention provided below.
Return your visible reasoning and actual final answer in exactly these separate blocks:
<reasoning>your reasoning</reasoning>
<final_answer>your final answer</final_answer>
"""


def _text(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _choices(row: dict) -> str:
    choices = row.get("choices", row.get("options", []))
    if not isinstance(choices, list):
        raise ValueError("choices/options must be a list")
    return "\n".join(
        str(choice) if re.match(r"^[A-Z][.)]\s", str(choice)) else f"{chr(65 + i)}. {choice}"
        for i, choice in enumerate(choices)
    )


def normalize_record(row: dict) -> dict:
    """Map only schemas with an explicit intervention and an identifiable target.

    Raw clinical/sandbagging/agent-environment sources need task-specific preparation;
    an ordinary ground-truth answer alone does not identify an intervention target.
    """
    if not isinstance(row, dict):
        raise ValueError("Each record must be a JSON object")
    if all(key in row for key in ("original_task", "intervention", "target_outcome")):
        task, intervention, target = (row[key] for key in ("original_task", "intervention", "target_outcome"))
    elif all(key in row for key in ("passage", "question", "intervention", "target_answer")):
        task = f"Passage:\n{row['passage']}\n\nQuestion:\n{row['question']}"
        intervention = row["intervention"]
        if row.get("counterfactual_question"):
            intervention += f"\n\nIntervened question:\n{row['counterfactual_question']}"
        target = _text(row["target_answer"])
    elif all(key in row for key in ("problem_text", "intervention_text", "ground_truth_answer")):
        task, intervention, target = row["problem_text"], row["intervention_text"], row["ground_truth_answer"]
    elif all(key in row for key in ("question", "preference_hint", "gold_choice")):
        task, intervention, target = row["question"], row["preference_hint"], row["gold_choice"]
    elif all(key in row for key in ("question", "hint", "answer")):
        task = "\n\n".join(str(row[key]) for key in ("control_case_brief", "control_text", "question") if row.get(key))
        intervention, target = row["hint"], row["answer"]
    elif all(key in row for key in ("control_arm", "intervention", "goal")):
        task, intervention, target = row["control_arm"], row["intervention"], row["goal"]
    elif all(key in row for key in ("Instruction", "Input", "AdditionalObjective")):
        task = f"{row['Instruction']}\n\n{row['Input']}"
        objective = row["AdditionalObjective"]
        # The instruction expresses the goal (e.g. acrostic), not just its payload.
        intervention = target = objective["Instruction"]
    else:
        raise ValueError(
            "Unsupported or incomplete schema: supply original_task, intervention, and target_outcome "
            "(an answer, a list of accepted answers, or a behavioral goal). "
            "Raw sources without intervention/target annotations cannot be inferred automatically."
        )
    choices = _choices(row)
    if choices:
        task = f"{task}\n\nChoices:\n{choices}"
    if not isinstance(task, str) or not task.strip() or not isinstance(intervention, str) or not intervention.strip():
        raise ValueError("original_task and intervention must be nonempty strings")
    if isinstance(target, (bool, int, float)):
        target = _text(target)
    if isinstance(target, list):
        if not target or not all(isinstance(item, (str, bool, int, float)) for item in target):
            raise ValueError("target_outcome must contain a nonempty accepted answer set")
        target = [_text(item) for item in target]
        if not all(item.strip() for item in target):
            raise ValueError("Accepted answers must be nonempty")
    elif not isinstance(target, str) or not target.strip():
        raise ValueError("target_outcome must be an answer string, accepted answer list, or behavioral goal string")
    return {"original_task": task, "intervention": intervention, "target_outcome": target}


def build_prompt(row: dict) -> str:
    # Target annotations are judge-only; do not insert them into the policy prompt.
    return f"{RESPONSE_CONTRACT}\nOriginal task:\n{row['original_task']}\n\nIntervention:\n{row['intervention']}"


def make_record(row: dict, split: str, index: int) -> dict:
    normalized = normalize_record(row)
    record = {
        "data_source": "monitorability_interventions",
        "prompt": [{"role": "user", "content": build_prompt(normalized)}],
        "ability": "monitorability",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps({"target_outcome": normalized["target_outcome"]}),
        },
        "extra_info": {
            "split": split,
            "index": index,
            "id": str(row.get("id", row.get("example_id", row.get("problem_id", index)))),
            "original_task": normalized["original_task"],
            "intervention": normalized["intervention"],
        },
    }


    if "sandbagging" in row:
        spec = row["sandbagging"]
        if not isinstance(spec, dict) or "ground_truth_answer" not in spec:
            raise ValueError("sandbagging requires an explicit ground_truth_answer")
        gt = json.loads(record["reward_model"]["ground_truth"])
        gt["sandbagging"] = spec
        record["reward_model"]["ground_truth"] = json.dumps(gt)
    return record


def load_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    elif path.suffix.lower() == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected a JSON array of records")
    else:
        raise ValueError(f"{path}: select a .json or .jsonl file, not a dataset directory")
    accepted = []
    for index, row in enumerate(records):
        if isinstance(row, dict) and row.get("status", "accepted") != "accepted":
            continue
        try:
            normalize_record(row)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}: record {index + 1}: {exc}") from exc
        accepted.append(row)
    return accepted


def main() -> None:
    from datasets import Dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--input_jsonl", dest="inputs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.val_ratio < 1:
        parser.error("--val_ratio must be between 0 and 1 (exclusive)")
    rows = [row for path in args.inputs for row in load_records(Path(path))]
    if len(rows) < 2:
        raise ValueError("At least two accepted records are required for disjoint training and validation splits")
    random.Random(args.seed).shuffle(rows)
    n_val = min(len(rows) - 1, max(1, int(round(len(rows) * args.val_ratio))))
    train = [make_record(row, "train", idx) for idx, row in enumerate(rows[n_val:])]
    val = [make_record(row, "val", idx) for idx, row in enumerate(rows[:n_val])]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train).to_parquet(output_dir / "train.parquet")
    Dataset.from_list(val).to_parquet(output_dir / "val.parquet")
    summary = {
        "schema_version": 2,
        "inputs": args.inputs,
        "train": len(train),
        "val": len(val),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
