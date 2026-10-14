"""Exclusive output bundles and content-addressed input provenance."""
import json
from pathlib import Path
from .config import sha,require

def read(path):return json.loads(Path(path).read_text())
def rows(path):
    with Path(path).open() as f:return [json.loads(line) for line in f if line.strip()]
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def new_directory(path):
    path=Path(path).resolve()
    package=Path(__file__).resolve().parent
    require(not path.is_relative_to(package),'Outputs cannot be inside installed package assets/code')
    path.mkdir(parents=True,exist_ok=False)
    return path
def seal(root):
    root=Path(root)
    files={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'COMPLETE.json'}
    write(root/'COMPLETE.json',{'files_sha256':files})
    return files
def verify(root):
    root=Path(root).resolve();lock=read(root/'COMPLETE.json')
    require(lock['files_sha256'],'Empty completion marker')
    for name,h in lock['files_sha256'].items():
        p=root/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and p.resolve().is_relative_to(root) and not p.is_symlink(),'Unsafe artifact path')
        require(p.is_file() and sha(p)==h,'Artifact changed or missing: '+name)
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=root/'COMPLETE.json'}
    require(actual==set(lock['files_sha256']),'Artifact file set changed')
    return lock
