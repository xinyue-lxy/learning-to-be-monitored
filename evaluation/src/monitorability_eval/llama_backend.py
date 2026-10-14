"""Single-query Llama monitoring with official-template token parity."""
import argparse
import copy
import json
from pathlib import Path
from .config import require,monitor_config,sha
from .artifacts import rows
from .environment import require_generation_environment

def token_ids(query,tokenizer,cfg,format_prompts):
    rendered=format_prompts([query['messages']],tokenizer,cfg)[0]
    direct=tokenizer.apply_chat_template(query['messages'],tokenize=False,add_generation_prompt=True)
    require(rendered==direct,'Formatter changed monitor messages')
    ids=list(tokenizer.encode(rendered,add_special_tokens=False))
    native=list(tokenizer.apply_chat_template(query['messages'],tokenize=True,add_generation_prompt=True))
    require(ids==native,'Official-template token parity failed')
    require(len(ids)+cfg.max_tokens<=cfg.max_model_len,'Context overflow; truncation is forbidden')
    return ids

def run(input_path,output_path,seed):
    require_generation_environment()
    from .engine.pipeline.generation.run import GenerationConfig,format_prompts,create_sampling_params
    from vllm import LLM
    settings=monitor_config('llama33_70b',seed,'cue_condition.health')
    cfg=GenerationConfig(model=settings['model_name'],revision=settings['revision'],temperature=settings['temperature'],
        top_p=settings['top_p'],seed=seed,n=settings['rollout_number'],max_tokens=settings['max_tokens'],
        max_model_len=settings['max_model_len'],tp=1,rope_scaling=None,chat_template_kwargs={},output_format='auto')
    kwargs={k:v for k,v in settings['serving'].items() if k!='prompt_batch_size'}
    kwargs.update(model=cfg.model,revision=cfg.revision,tokenizer_revision=cfg.revision,dtype='bfloat16',
                  quantization=None,trust_remote_code=False,max_model_len=cfg.max_model_len,seed=seed)
    queries=rows(input_path)
    if not queries:
        Path(output_path).touch(exist_ok=False);return
    model=LLM(**kwargs)
    require(str(model.llm_engine.model_config.dtype)=='torch.bfloat16','Unexpected monitor dtype')
    tokenizer=model.get_tokenizer();sampling=create_sampling_params(cfg)
    with Path(output_path).open('x') as f:
        for query in queries:
            ids=token_ids(query,tokenizer,cfg,format_prompts)
            outputs=model.generate([{'prompt_token_ids':ids}],sampling,use_tqdm=False)
            require(len(outputs)==1 and list(outputs[0].prompt_token_ids)==ids,'Engine changed prompt tokens or request coverage')
            items=outputs[0].outputs
            require(len(items)==4 and {x.index for x in items}==set(range(4)),'Incomplete monitor rollouts')
            row=copy.deepcopy(query)
            row['response']=[x.text for x in items]
            row['response_serialization']=[{'finish_reason':x.finish_reason,'stop_reason':x.stop_reason,
                                            'generated_tokens':len(x.token_ids),'prompt_tokens':len(ids)} for x in items]
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();run(a.input,a.output,a.seed)
if __name__=='__main__':main()
