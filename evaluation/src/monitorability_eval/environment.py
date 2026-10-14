"""Explicit cache locations and compute allocation checks; no environment install."""
import importlib.metadata
import os
from pathlib import Path
from .config import profiles,require

CACHE_NAMES=('HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','VLLM_CACHE_ROOT',
             'TIKTOKEN_RS_CACHE_DIR','TIKTOKEN_CACHE_DIR','TORCH_HOME','TRITON_CACHE_DIR',
             'CUDA_CACHE_PATH','XDG_CACHE_HOME')

def cache_root():
    value=os.environ.get('MONITORABILITY_CACHE_DIR')
    require(value is not None and Path(value).is_absolute(),'Set MONITORABILITY_CACHE_DIR to an absolute shared/scratch cache directory')
    p=Path(value).resolve()
    require(p!=Path('/') and not p.is_relative_to(Path.home().resolve()),'Cache must be outside the home directory and not the filesystem root')
    return p

def configure_environment():
    root=cache_root()
    sub={'HF_HOME':'huggingface','HF_HUB_CACHE':'huggingface/hub','HUGGINGFACE_HUB_CACHE':'huggingface/hub',
         'VLLM_CACHE_ROOT':'vllm','TIKTOKEN_RS_CACHE_DIR':'tiktoken','TIKTOKEN_CACHE_DIR':'tiktoken',
         'TORCH_HOME':'torch','TRITON_CACHE_DIR':'triton','CUDA_CACHE_PATH':'cuda','XDG_CACHE_HOME':'xdg'}
    for name,folder in sub.items():
        if name in os.environ:
            require(Path(os.environ[name]).resolve().is_relative_to(root),'Conflicting cache path: '+name)
        else:os.environ[name]=str(root/folder)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
                      WANDB_MODE='disabled',WANDB_DISABLED='true',PYTHONDONTWRITEBYTECODE='1')
    return {name:os.environ[name] for name in CACHE_NAMES}

def require_generation_environment():
    require(os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURMD_NODENAME') or os.environ.get('SLURM_JOB_NODELIST')),
            'Run generation inside a Slurm compute allocation, never on a login node')
    root=cache_root()
    for name in CACHE_NAMES:
        require(os.environ.get(name) and Path(os.environ[name]).resolve().is_relative_to(root),'Unsafe/missing cache: '+name)
    require(os.environ.get('HF_HUB_OFFLINE')=='1','Only pre-cached model snapshots are permitted')
    require(os.environ.get('WANDB_MODE')=='disabled','W&B must be disabled')

def runtime_versions(strict=True):
    expected=profiles()['runtime_versions'];actual={}
    for name,want in expected.items():
        try:actual[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:actual[name]=None
        if strict:require(actual[name] is not None and actual[name].split('+')[0]==want,'Runtime version mismatch: '+name)
    return actual

def require_hardware(count):
    require_generation_environment()
    require(os.environ.get('SLURM_JOB_PARTITION') in ('gpu_h200','gpu_b200'),'Use a non-preemptible H200/B200 partition')
    import torch
    require(torch.cuda.device_count()==count,'Allocation GPU count differs from the selected profile')
    require(all(any(name in torch.cuda.get_device_name(i) for name in ('H200','B200')) for i in range(count)),
            'Only typed H200/B200 GPUs are supported by this execution profile')
