"""General evaluation CLI. Planning is the default; inference is explicit."""
import argparse
import json
from pathlib import Path

from . import artifacts
from .analysis import compare
from .config import profiles,policy,verify_assets
from .runner import plan_task,run_task,plan_monitor,run_monitor,validate_checkpoint

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('inspect',help='List pinned profiles, policy and dataset integrity')
    checkpoint=sub.add_parser('checkpoint',help='Verify a local unmerged checkpoint package without loading weights')
    checkpoint.add_argument('--model',choices=profiles()['task_models'],required=True)
    checkpoint.add_argument('--path',required=True)
    checkpoint.add_argument('--hash-payloads',action='store_true')
    task=sub.add_parser('task',help='Plan or generate task rollouts and corrected Y')
    task.add_argument('--model',choices=profiles()['task_models'],required=True)
    task.add_argument('--task',choices=policy()['tasks'],required=True)
    task.add_argument('--seed',type=int,required=True)
    task.add_argument('--verifier-seed',type=int,required=True)
    task.add_argument('--label',required=True)
    task.add_argument('--adapter',help='Checkpoint package containing adapter/, tokenizer/, checkpoint_manifest.json')
    task.add_argument('--sample-limit',type=int,help='Diagnostic canary only, never silently included in formal analysis')
    task.add_argument('--output',required=True)
    task.add_argument('--execute',action='store_true')
    monitor=sub.add_parser('monitor',help='Plan or replay a monitor on an immutable task/Y bundle')
    monitor.add_argument('--source',required=True)
    monitor.add_argument('--monitor',choices=profiles()['monitors'],required=True)
    monitor.add_argument('--seed',type=int,required=True)
    monitor.add_argument('--output',required=True)
    monitor.add_argument('--execute',action='store_true')
    analysis=sub.add_parser('analyze',help='Instance-first bounds on native or common support')
    analysis.add_argument('--bundles',nargs='+',required=True,help='Sealed monitor output directories, one monitor/model family per invocation')
    analysis.add_argument('--base',default='base')
    analysis.add_argument('--candidates',nargs='+',required=True)
    analysis.add_argument('--seeds',nargs='+',type=int,required=True)
    analysis.add_argument('--tasks',nargs='+',choices=policy()['tasks'],help='Omit for the complete task set')
    analysis.add_argument('--support',choices=['native','common','trajectory_common'],required=True)
    analysis.add_argument('--allow-diagnostic',action='store_true')
    analysis.add_argument('--output',help='Optional new JSON file; existing files are never overwritten')
    return p

def main(argv=None):
    a=parser().parse_args(argv)
    try:
        if a.command=='inspect':result={'profiles':profiles(),'policy':policy(),'integrity':verify_assets()}
        elif a.command=='checkpoint':result=validate_checkpoint(a.path,a.model,a.hash_payloads)
        elif a.command=='task':result=run_task(plan_task(a.model,a.task,a.seed,a.verifier_seed,a.label,a.adapter,a.sample_limit),a.output,a.execute)
        elif a.command=='monitor':result=run_monitor(plan_monitor(a.source,a.monitor,a.seed),a.output,a.execute)
        else:
            cells=[]
            for root in a.bundles:
                artifacts.verify(root)
                cells.append(artifacts.read(Path(root)/'cell.json'))
            result=compare(cells,base=a.base,candidates=a.candidates,seeds=a.seeds,tasks=a.tasks,
                           support=a.support,allow_diagnostic=a.allow_diagnostic)
            if a.output:artifacts.write(a.output,result)
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,OSError,KeyError) as exc:
        raise SystemExit(f'Evaluation refused: {exc}') from exc

if __name__=='__main__':main()
