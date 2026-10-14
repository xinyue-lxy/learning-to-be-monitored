"""Convert user-supplied category records to paired control/intervention rows.

Run separately for training and validation files; this adapter does not choose a split.
"""
import argparse
import json
from pathlib import Path

from .context import CATEGORY_GUIDES, PUBLIC_FIELDS, student_messages


def make_pair(record, index):
    category = record['category']
    if category not in CATEGORY_GUIDES:
        raise ValueError(f'Unsupported category: {category}')
    factor = record['intervention']
    factor_type = category.split('.')[0].split('_', 1)[1]
    if factor['type'] != factor_type or not factor['text'].strip():
        raise ValueError('Category and intervention must have matching types and nonempty text')
    if not any(record.get(key) for key in PUBLIC_FIELDS):
        raise ValueError('At least one public task field is required')
    task_id = str(record.get('id', index))
    rows = []
    for arm in (0, 1):
        metadata = {
            'task_id': task_id, 'instance_id': task_id,
            'category': category, 'arm': arm, 'factor_type': factor_type,
            'factor_text': factor['text'], 'registered_answer': record['ground_truth_answer'],
        }
        rows.append({
            'data_source': category, 'agent_name': 'opsd_agent',
            'prompt': student_messages(record, arm), 'ability': 'reasoning',
            'reward_model': {'style': 'rule', 'ground_truth': ''}, 'extra_info': metadata,
        })
    return rows


def main():
    from datasets import Dataset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    rows = [row for i, record in enumerate(records) for row in make_pair(record, i)]
    if not rows:
        raise ValueError('Input is empty')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(args.output)


if __name__ == '__main__':
    main()
