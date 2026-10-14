"""Data loading and configuration for full-assistant supervised fine-tuning."""
from pathlib import Path
import hashlib
import json
import math
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def read_rows(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, set):
        return [json_safe(v) for v in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json_new(path, value):
    payload = json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + '\n'
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(payload)


def load_config(path):
    import yaml
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    require(config['model_family'] in ('qwen3_5', 'mistral3'), 'Unsupported model family')
    require(config['epochs'] > 0 and config['max_sequence_length'] > 0, 'Invalid epoch or sequence limit')
    opt = config['optimizer']
    require(opt['effective_batch_size'] > 0 and opt['microbatch_size'] > 0, 'Invalid batch size')
    require(opt['learning_rate'] > 0 and opt['gradient_clip_norm'] > 0, 'Invalid optimizer settings')
    require(0 <= config['scheduler']['warmup_ratio'] <= 1, 'Invalid warmup ratio')
    require(0 <= config['scheduler']['minimum_learning_rate_ratio'] <= 1, 'Invalid minimum LR ratio')
    lora = config['lora']
    require(lora['rank'] > 0 and lora['alpha'] > 0 and 0 <= lora['dropout'] < 1, 'Invalid LoRA settings')
    lora.setdefault('bias', 'none')
    # Default targets follow each supported architecture, not an experiment run.
    if 'target_modules' not in lora:
        lora['target_modules'] = (
            r'model\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)'
            if config['model_family'] == 'mistral3' else
            ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj_qkv', 'in_proj_z', 'in_proj_a', 'in_proj_b', 'out_proj']
        )
    require(lora['bias'] == 'none', 'Only adapter weights are trained')
    return config


def validate_row(row):
    messages, payload = row['ordinary_student_messages'], row['target_assistant_content']
    require([m['role'] for m in messages] == ['system', 'user'], 'Expected public system and user messages')
    require(all(isinstance(m['content'], str) and m['content'].strip() for m in messages), 'Empty public message')
    require(isinstance(payload, str), 'Assistant target must be a string')
    match = re.fullmatch(r'\s*<reasoning>(.*?)</reasoning>\s*<answer>(.*?)</answer>\s*', payload, re.DOTALL)
    require(match is not None and match[1].strip() and match[2].strip(), 'Invalid or empty assistant blocks')
    for name in ('reasoning', 'answer'):
        require(payload.count('<'+name+'>') == payload.count('</'+name+'>') == 1, 'Duplicate assistant blocks')


def epoch_plan(train, epochs, batch_size, seed=42):
    require(epochs > 0 and batch_size > 0, 'Invalid epoch or batch size')
    ids = [r['request_id'] for r in train]
    require(ids and len(set(ids)) == len(ids), 'Empty or duplicate training requests')
    plans = []
    for epoch in range(1, epochs + 1):
        order = ids if epoch == 1 else sorted(ids, key=lambda rid: (sha(f'{seed}:epoch{epoch}:{rid}'), rid))
        batches = [order[i:i+batch_size] for i in range(0, len(order), batch_size)]
        plans.append(dict(epoch=epoch, batches=batches, last_optimizer_step=epoch*len(batches)))
    return plans


def load_data(config, train_path, validation_path=None):
    def load(path, split):
        rows = read_rows(path) if path is not None else []
        for row in rows:
            validate_row(row)
            require(row.get('split', split) == split, 'Split mismatch')
            require(row.get('arm', 'intervention') == 'intervention', 'Expected intervention-arm targets')
            payload = row['target_assistant_content']
            match = re.fullmatch(r'\s*<reasoning>(.*?)</reasoning>\s*<answer>(.*?)</answer>\s*', payload, re.DOTALL)
            row['target_cot'], row['target_answer'] = match[1].strip(), match[2].strip()
            row.setdefault('request_id', sha(canonical(row['ordinary_student_messages'])+payload))
            row['request_id'] = str(row['request_id'])
            row.setdefault('category', 'uncategorized')
        require(len({r['request_id'] for r in rows}) == len(rows), 'Duplicate request ID')
        return rows
    train, validation = load(train_path, 'train'), load(validation_path, 'validation')
    require(train, 'Training data is empty')
    for field in ('request_id', 'record_id', 'content_group_id'):
        require(not {r[field] for r in train if field in r} & {r[field] for r in validation if field in r},
                'Train/validation overlap: '+field)
    require(not {canonical(r['ordinary_student_messages']) for r in train} &
            {canonical(r['ordinary_student_messages']) for r in validation}, 'Public prompt crosses splits')
    plans = epoch_plan(train, config['epochs'], config['optimizer']['effective_batch_size'], config['seed'])
    config['scheduler']['total_optimizer_steps'] = plans[-1]['last_optimizer_step']
    config['scheduler']['warmup_steps'] = math.ceil(config['scheduler']['warmup_ratio'] * plans[-1]['last_optimizer_step'])
    return train, validation, plans
