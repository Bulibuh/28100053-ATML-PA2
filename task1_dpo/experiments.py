"""Complete DPO workflows, leaving the released scaffolds intact."""
from __future__ import annotations
import argparse
import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path, prompt_messages_from_preference
from common.generation import response_sequence_logprobs
from common.models import reference_mode, load_policy, load_tokenizer, load_reward_model
from common.logging_utils import set_seed
from common.metrics import word_limit_compliance
from common.experiment_utils import (Run, dump, jsonl, deterministic_policy, norm_and_step, generate_evaluation,
                                      require_accelerator, release_memory, identity)
from task1_dpo.train import prepare_dpo_run, make_collate
from task1_dpo.dpo import dpo_loss


def train(config='configs/dpo.yaml', name='standard', dataset=None, output=None, beta=None, max_examples=None, max_sequence_length=None):
    b=prepare_dpo_run(config,dataset,beta,max_examples)
    cfg=dict(b['cfg'], beta=b['beta'], training_dataset=dataset or b['cfg']['paths']['dpo_standard_train'], max_examples=max_examples)
    model, optimizer, loader=b['model'],b['optimizer'],b['loader']
    if max_sequence_length is not None:
        cfg['released_max_sequence_length']=cfg['max_sequence_length']
        cfg['max_sequence_length']=int(max_sequence_length)
        loader.collate_fn=make_collate(b['tokenizer'],int(max_sequence_length))
    if not len(loader): raise ValueError('Empty DPO dataset')
    out=repo_path(output or (cfg['standard_output'] if name=='standard' else f'outputs/task1_dpo/{name}'))
    if out.exists() and (out/'adapter_config.json').exists():raise FileExistsError(out)
    run=Run(cfg['results_dir'],name,cfg,b['rows'])
    device=next(model.parameters()).device
    accum=int(cfg['grad_accum_steps']); steps=0; seen=0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(cfg['epochs'])):
        losses=[]; accuracy=[]
        for i,(chosen,rejected) in enumerate(loader):
            chosen={k:v.to(device) for k,v in chosen.items()}; rejected={k:v.to(device) for k,v in rejected.items()}
            with torch.no_grad(), reference_mode(model):
                rc,_,_=response_sequence_logprobs(model,chosen); rr,_,_=response_sequence_logprobs(model,rejected)
            with torch.enable_grad():
                pc,_,_=response_sequence_logprobs(model,chosen); pr,_,_=response_sequence_logprobs(model,rejected)
                loss,diag=dpo_loss(pc,pr,rc,rr,b['beta'])
                # Last partial accumulation window averages its actual microbatches.
                window=min(accum,len(loader)-(i//accum)*accum)
                (loss/window).backward()
            losses.append(float(loss.detach())); accuracy.append(float(diag['preference_accuracy'])); seen+=len(pc)
            if (i+1)%accum==0 or i+1==len(loader):
                norm=norm_and_step(model,optimizer,cfg['max_grad_norm']); steps+=1
                run.log(step=steps,epoch=epoch,examples_seen=seen,loss=float(np.mean(losses)),preference_accuracy=float(np.mean(accuracy)),gradient_norm=norm)
                losses=[]; accuracy=[]
    out.mkdir(parents=True,exist_ok=True); model.save_pretrained(out); b['tokenizer'].save_pretrained(out)
    run.finish(adapter=str(out),optimizer_steps=steps,examples_seen=seen)
    return str(out)


def evaluate(config='configs/dpo.yaml', adapter=None, name='standard', dataset=None, word_limits=False, max_sequence_length=None):
    cfg=load_yaml(config); set_seed(int(cfg['seed']))
    if max_sequence_length is not None:
        cfg['released_max_sequence_length']=cfg['max_sequence_length']
        cfg['max_sequence_length']=int(max_sequence_length)
    rows=read_jsonl(dataset or cfg['paths']['dpo_standard_eval'])
    if not rows:raise ValueError('Empty DPO evaluation set')
    if dataset==cfg['paths']['dpo_length_eval'] and any('length_stratum' not in row for row in rows):
        raise ValueError('The fixed length-stratified file must contain course length_stratum labels')
    tok=load_tokenizer(cfg['base_model']); policy=load_policy(cfg,adapter_path=adapter,trainable=False)
    collate=make_collate(tok,int(cfg['max_sequence_length'])); preference=[]
    device=next(policy.parameters()).device
    with torch.no_grad(),deterministic_policy(policy):
        for row in rows:
            c,r=collate([row]); c={k:v.to(device) for k,v in c.items()}; r={k:v.to(device) for k,v in r.items()}
            pc,_,_=response_sequence_logprobs(policy,c); pr,_,_=response_sequence_logprobs(policy,r)
            with reference_mode(policy):
                rc,_,_=response_sequence_logprobs(policy,c); rr,_,_=response_sequence_logprobs(policy,r)
            loss,diag=dpo_loss(pc,pr,rc,rr,float(cfg['beta']))
            preference.append({**identity(row),'stratum':row.get('length_stratum',row.get('stratum','unspecified')),
                               'dpo_loss':float(loss),'correct':float(diag['preference_accuracy']),'margin':float((pc-rc)-(pr-rr))})
    generation_rows=[{**identity(r),'messages':prompt_messages_from_preference(r)} for r in rows]
    reward=load_reward_model(cfg)
    metrics,records=generate_evaluation(cfg,generation_rows,tok,policy,reward,int(cfg['max_generation_tokens']))
    metrics.update(dpo_loss=float(np.mean([r['dpo_loss'] for r in preference])),preference_accuracy=float(np.mean([r['correct'] for r in preference])))
    for stratum in sorted(set(r['stratum'] for r in preference)):
        subset=[r for r in preference if r['stratum']==stratum]
        metrics.setdefault('strata',{})[stratum]={'n':len(subset),'preference_accuracy':float(np.mean([r['correct'] for r in subset]))}
    if word_limits:
        wr=read_jsonl(cfg['paths']['word_limit_prompts'])
        wm,examples=generate_evaluation(cfg,wr,tok,policy,reward,int(cfg['max_generation_tokens']))
        for row,rec in zip(wr,examples):
            text=str(row.get('prompt',row.get('messages',[{}])[-1].get('content','')))
            # The fixed file supplies explicit word_limit alongside its prompt.
            if row.get('word_limit') is not None:
                from common.metrics import word_count
                rec['word_limit_compliance']=float(word_count(rec['response'])<=int(row['word_limit']))
            else:rec['word_limit_compliance']=word_limit_compliance(text,rec['response'])
        valid=[r['word_limit_compliance'] for r in examples if r['word_limit_compliance'] is not None]
        if not valid:raise ValueError('No parsed word limits in fixed file')
        wm['word_limit_compliance']=float(np.mean(valid)); metrics['word_limits']=wm
        jsonl(f"{cfg['results_dir']}/{name}/word_limit_examples.jsonl",examples)
    directory=f"{cfg['results_dir']}/{name}"
    jsonl(f'{directory}/preferences.jsonl',preference);jsonl(f'{directory}/generations.jsonl',records)
    dump(f'{directory}/metrics.json',{'config':cfg,'adapter':adapter,'dataset':dataset or cfg['paths']['dpo_standard_eval'],**metrics})
    print(metrics)
    return metrics


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',default='configs/dpo.yaml');ap.add_argument('--mode',choices=['standard','beta','length','evaluate','all'],default='all')
    ap.add_argument('--adapter');ap.add_argument('--name',default='standard')
    ap.add_argument('--max-sequence-length',type=int,help='Explicit prompt-preserving override; recorded in every run')
    args=ap.parse_args();cfg=load_yaml(args.config)
    if args.mode!='evaluate': require_accelerator()
    if args.mode in ('standard','all'):
        adapter=train(args.config,max_sequence_length=args.max_sequence_length); release_memory();evaluate(args.config,adapter,max_sequence_length=args.max_sequence_length);release_memory()
    if args.mode in ('beta','all'):
        for beta in cfg['betas']:
            name=f'beta_{beta:.2f}';adapter=train(args.config,name,beta=beta,max_examples=cfg['short_ablation_examples'],max_sequence_length=args.max_sequence_length);release_memory()
            # Evaluate the objective at the training beta without altering the supplied config.
            from common.experiment_utils import dump
            local=repo_path(f"outputs/task1_dpo/{name}/evaluation_config.json")
            dump(local,dict(cfg,beta=float(beta)))
            evaluate(str(local),adapter,name,max_sequence_length=args.max_sequence_length);release_memory()
    if args.mode in ('length','all'):
        adapter=train(args.config,'length_balanced',cfg['paths']['dpo_length_train'],cfg['length_output'],max_sequence_length=args.max_sequence_length);release_memory()
        for name,path in [('standard_length',cfg['standard_output']),('balanced_length',adapter)]:
            evaluate(args.config,path,name,cfg['paths']['dpo_length_eval'],True,args.max_sequence_length);release_memory()
    if args.mode=='evaluate':evaluate(args.config,args.adapter or cfg['standard_output'],args.name,max_sequence_length=args.max_sequence_length)


if __name__=='__main__':main()
