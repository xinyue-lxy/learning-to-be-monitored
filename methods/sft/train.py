"""Single-GPU LoRA SFT on user-supplied offline teacher targets."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import time

from .common import (canonical, json_safe,
                     load_config, load_data, require, write_json_new)
from .core import (adapter_gradient_norm, collate_right_padded, cosine_learning_rate,
                   load_qwen35_text_lora_model, save_checkpoint_atomic,
                   assistant_ce_per_trajectory, assistant_ce_per_trajectory_vector)


def load_model(config, snapshot):
    import torch
    if config['model_family'] == 'qwen3_5':
        local = json.loads(json.dumps(config))
        local.setdefault('model', {})['snapshot_path'] = str(snapshot)
        model, summary = load_qwen35_text_lora_model(local, device='cuda:0')
        summary.pop('trainable_parameter_names',None)
    else:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import Mistral3ForConditionalGeneration
        model, loading = Mistral3ForConditionalGeneration.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'':'cuda:0'},
            output_loading_info=True)
        for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'):
            require(not loading.get(key), 'Incomplete/incompatible Mistral base load: '+key)
        require(model.config.model_type == 'mistral3',
                'Wrong Mistral architecture')
        lora = config['lora']
        selected = [n for n,_ in model.named_modules() if re.fullmatch(lora['target_modules'],n)]
        require(selected, 'No matching Mistral attention modules')
        model.config.use_cache = False
        model.config.text_config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        model = get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=lora['rank'],
            lora_alpha=lora['alpha'],lora_dropout=lora['dropout'],bias='none',target_modules=lora['target_modules']))
        trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
        require(len(trainable) == 2*len(selected) and all(re.fullmatch(r'base_model\.model\.'+lora['target_modules']+
            r'\.lora_[AB]\.default\.weight',n) for n,_ in trainable),'Unexpected Mistral trainable scope')
        summary = dict(trainable_parameter_count=sum(p.numel() for _,p in trainable),
            lora_modules=len(selected),base_loading_info=json_safe({k:loading.get(k,[]) for k in
                ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')}))
    require(all(p.device.type == 'cuda' for p in model.parameters()), 'This trainer requires model parameters on its CUDA device')
    return model, summary


def optimizer_step(model, optimizer, examples, microbatch_size, clip, pad_token_id, device):
    import torch
    require(examples and microbatch_size > 0, 'Empty batch or invalid microbatch')
    optimizer.zero_grad(set_to_none=True)
    use_cuda = str(device).startswith('cuda')
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    aggregate, supervised = 0.0, 0
    model.train()
    for start in range(0,len(examples),microbatch_size):
        chunk = examples[start:start+microbatch_size]
        weight = len(chunk)/len(examples)
        batch = collate_right_padded([r['target_path'] for r in chunk],pad_token_id=pad_token_id,device=device)
        loss, counts = assistant_ce_per_trajectory(model,batch)
        require(bool(torch.isfinite(loss)), 'Non-finite assistant CE')
        (weight*loss).backward()
        aggregate += weight*float(loss.detach())
        supervised += sum(counts)
    norm = adapter_gradient_norm(model)
    require(math.isfinite(norm) and norm > 0, 'Non-finite or zero gradient')
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],clip,error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if use_cuda:
        torch.cuda.synchronize()
    return dict(full_assistant_ce=aggregate,gradient_norm_before_clip=norm,
        supervised_assistant_token_count=supervised,logical_batch_size=len(examples),
        gradient_accumulation_chunks=math.ceil(len(examples)/microbatch_size),
        step_seconds=time.perf_counter()-started,
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()) if use_cuda else 0)


def validation_ce(model, rows, pad_token_id, device):
    import torch
    from collections import defaultdict
    require(rows, 'Empty validation set')
    by_category = defaultdict(list)
    values = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0,len(rows),2):
                chunk = rows[start:start+2]
                batch = collate_right_padded([r['target_path'] for r in chunk],pad_token_id=pad_token_id,device=device)
                losses,_ = assistant_ce_per_trajectory_vector(model,batch)
                for row,value in zip(chunk,losses.tolist()):
                    require(math.isfinite(value), 'Non-finite validation loss')
                    values.append(value)
                    by_category[row['category']].append(value)
    finally:
        model.train(was_training)
    return dict(count=len(values),natural_trajectory_mean=sum(values)/len(values),
        by_category={k:dict(count=len(v),mean_ce=sum(v)/len(v)) for k,v in sorted(by_category.items())},
        category_macro_mean=sum(sum(v)/len(v) for v in by_category.values())/len(by_category),optimizer_used=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--model-path',required=True,type=Path,help='Local Hugging Face model snapshot')
    parser.add_argument('--output',required=True,type=Path,help='New output directory; never overwritten')
    parser.add_argument('--train-file', required=True, type=Path)
    parser.add_argument('--validation-file', type=Path)
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    os.environ['WANDB_MODE']='disabled'
    from .serialization import serialize_all
    config = load_config(args.config)
    snapshot = args.model_path.resolve()
    require(not args.output.exists(), 'Output exists; no overwrite or resume')
    train,validation,plans = load_data(config,args.train_file,args.validation_file)
    serializer,serialized = serialize_all(config,snapshot,train,validation)
    print(f"Prepared {len(train)+len(validation)} targets.",flush=True)
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expose exactly one CUDA GPU')
    gpu = torch.cuda.get_device_name(0)
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    report=dict(complete=False,config=config,gpu=gpu,steps=[],validation=[],checkpoints=[],
        checkpoint_selection_performed=False,logging_backend='local_json')
    try:
        model,report['parameter_summary']=load_model(config,snapshot)
        opt=config['optimizer']
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=opt['learning_rate'],
            betas=tuple(opt['betas']),eps=opt['epsilon'],weight_decay=opt['weight_decay'])
        by_id={r['request_id']:r for r in serialized['train']}
        if validation:
            initial=dict(optimizer_step=0,epoch=0,**validation_ce(model,serialized['validation'],serializer.pad_token_id,'cuda:0'))
            report['validation'].append(initial)
            write_json_new(args.output/'epoch-0-summary.json',initial)
        step=exposures=0
        with (args.output/'step_metrics.jsonl').open('x',encoding='utf-8') as log:
            for plan in plans:
                for ids in plan['batches']:
                    step+=1
                    schedule=config['scheduler']
                    lr=cosine_learning_rate(step,total_steps=schedule['total_optimizer_steps'],
                        warmup_step_count=schedule['warmup_steps'],peak_lr=opt['learning_rate'],
                        minimum_lr_ratio=schedule['minimum_learning_rate_ratio'])
                    for group in optimizer.param_groups:
                        group['lr']=lr
                    metrics=optimizer_step(model,optimizer,[by_id[rid] for rid in ids],opt['microbatch_size'],
                        opt['gradient_clip_norm'],serializer.pad_token_id,'cuda:0')
                    exposures+=len(ids)
                    metrics.update(optimizer_step=step,epoch=plan['epoch'],trained_trajectory_count=exposures,learning_rate=lr)
                    report['steps'].append(metrics)
                    log.write(canonical(metrics)+'\n')
                    log.flush()
                    print(canonical(metrics),flush=True)
                require(step == plan['last_optimizer_step'] and exposures == len(train)*plan['epoch'], 'Epoch accounting')
                if validation:
                    result=dict(optimizer_step=step,epoch=plan['epoch'],
                        **validation_ce(model,serialized['validation'],serializer.pad_token_id,'cuda:0'))
                    report['validation'].append(result)
                    write_json_new(args.output/f"epoch-{plan['epoch']}-summary.json",result)
                name=f'adapter-step-{step:06d}'
                checkpoint=save_checkpoint_atomic(args.output/'checkpoints'/name,
                    adapter_writer=lambda p:model.save_pretrained(p,safe_serialization=True),
                    tokenizer_writer=serializer.save_pretrained,
                    manifest=dict(model_family=config['model_family'],config=config,
                        optimizer_step=step,epoch=plan['epoch'],trained_trajectory_count=exposures,
                        objective='full_assistant_ce_no_kl',checkpoint_selection_performed=False,merged=False))
                report['checkpoints'].append(dict(directory=name,optimizer_step=step,resumable=checkpoint['resumable']))
        require(step == config['scheduler']['total_optimizer_steps'] and exposures == len(train)*config['epochs'], 'Incomplete training')
        report['complete']=True
    except BaseException as error:
        report['error_type']=type(error).__name__
        raise
    finally:
        report['wall_seconds']=time.perf_counter()-started
        write_json_new(args.output/'run_report.json',report)


if __name__ == '__main__':
    main()
