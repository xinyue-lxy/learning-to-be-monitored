"""Portable profiles: seeds, checkpoints and output paths are caller inputs."""
import copy
import hashlib
import json
from pathlib import Path

ASSETS=Path(__file__).parent/'assets'
def read_asset(name):return json.loads((ASSETS/name).read_text())
def dataset_manifest():return read_asset('datasets.json')
def policy():return read_asset('policy.json')
def profiles():return read_asset('profiles.json')
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)

def seed_value(value):
    require(type(value) is int and 0<=value<2**32,'Supply an integer seed in [0, 2**32)')
    return value

def fingerprint():
    root=Path(__file__).parent
    files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
           if p.is_file() and p.suffix in ('.py','.json','.jsonl','.yaml') and '__pycache__' not in p.parts}
    return digest(files)

def verify_assets():
    for name,h in read_asset('integrity.json').items():
        require(sha(ASSETS/name)==h,'Frozen asset changed: '+name)
    for task,spec in dataset_manifest().items():
        require(sha(ASSETS/spec['path'])==spec['sha256'],'Dataset changed: '+task)
    return {'datasets':len(dataset_manifest()),'policy_fingerprint':fingerprint()}

def task_config(model,seed,verifier_seed):
    p=profiles();require(model in p['task_models'],'Unknown task model')
    t=copy.deepcopy(p['task_models'][model]);t['seed']=seed_value(seed)
    v=copy.deepcopy(p['verifier']);v['seed']=seed_value(verifier_seed)
    return {'evaluated_llm':t,'verification_llm':v,'monitor_llm':monitor_config('qwen3_32b',seed,'cue_condition.health')}

def monitor_config(name,seed,task):
    p=profiles();require(name in p['monitors'],'Unknown monitor')
    require(task in policy()['tasks'],'Unknown task')
    m=copy.deepcopy(p['monitors'][name]);m['seed']=seed_value(seed)
    m['forced_final_choices']='AB' if task.startswith('cue_answer.') else 'ABC'
    return m
