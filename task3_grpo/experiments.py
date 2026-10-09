"""GRPO continuation, fixed-budget group diagnostics and normalization forks."""
from __future__ import annotations
import argparse
from collections import defaultdict
import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path, prompt_messages
from common.generation import batch_generate, score_reward_pairs
from common.models import load_policy, load_tokenizer, load_reward_model
from common.logging_utils import set_seed
from common.metrics import masked_mean, sampled_kl
from common.experiment_utils import (Run, dump, jsonl, deterministic_policy, rollout_stats, norm_and_step,
    generate_evaluation, require_accelerator, release_memory, identity, trainable_rollout)
from task3_grpo.continue_train import prepare_grpo_continuation
from task3_grpo.analyze_group_size import load_k8_cache
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences


def group_statistics(rewards,group_ids,tolerance=1e-6):
    stds=[float(rewards[group_ids==g].std(unbiased=False)) for g in torch.unique(group_ids)]
    advantages=group_relative_advantages(rewards,group_ids,tolerance)
    return {'reward_std':float(np.mean(stds)),'uninformative_fraction':float(np.mean(np.array(stds)<=tolerance)),
            'informative_fraction':float(np.mean(np.array(stds)>tolerance)),
            'advantage_variance':float(advantages.var(unbiased=False)), 'groups':len(stds)}


def length_gradient_statistics(lp,loss,mask):
    """dL/d log π: direct token gradient allocation, unaffected by model size."""
    gradient=torch.autograd.grad(loss,lp,retain_graph=True)[0].detach()
    records=[]
    for i in range(lp.shape[0]):
        n=int(mask[i].sum())
        records.append({'tokens':n,'logprob_gradient_l1':float((gradient[i].abs()*mask[i]).sum()),
                        'logprob_gradient_per_token':float((gradient[i].abs()*mask[i]).sum()/max(n,1))})
    return records


def train(config='configs/grpo.yaml',name='standard',updates=None,loss_type='grpo'):
    b=prepare_grpo_continuation(config);cfg=dict(b['cfg'],loss_type=loss_type)
    if updates is not None:cfg['updates']=int(updates)
    cfg['policy_dropout_enabled_during_ratio_updates']=False
    cfg['allocated_response_token_budget']=int(cfg['updates'])*int(cfg['prompts_per_update'])*int(cfg['num_generations'])*int(cfg['max_completion_length'])
    policy=b['policy'];optimizer=b['optimizer'];rows=b['prompt_rows'];tok=b['tokenizer']
    if not rows:raise ValueError('Empty GRPO prompt pool')
    out=repo_path(f'outputs/task3_grpo/{name}')
    if (out/'adapter_config.json').exists():raise FileExistsError(out)
    run=Run(cfg['results_dir'],name,cfg,rows);traces=[];length_stats=[]
    for step in range(int(cfg['updates'])):
        p=int(cfg['prompts_per_update']);k=int(cfg['num_generations'])
        selected=[rows[(step*p+j)%len(rows)] for j in range(p)]
        expanded=[r for r in selected for _ in range(k)];prompts=[prompt_messages(r) for r in expanded]
        gen=batch_generate(policy,tok,prompts,int(cfg['max_prompt_length']),int(cfg['max_completion_length']),**cfg['generation'])
        gen=trainable_rollout(gen)
        original_mask=gen['response_mask'];mask=original_mask
        if cfg['mask_truncated_completions']:mask=mask_truncated_sequences(mask,gen['truncated'])
        gids=torch.arange(p,device=mask.device).repeat_interleave(k)
        with deterministic_policy(policy),torch.no_grad():
            old,_=rollout_stats(policy,gen);ref,_=rollout_stats(policy,gen,True)
            rewards=score_reward_pairs(b['reward_model'],b['reward_tokenizer'],prompts,gen['responses'])
            advantages=group_relative_advantages(rewards,gids).detach()
        losses=[];norms=[]
        for epoch in range(int(cfg['policy_epochs'])):
            optimizer.zero_grad(set_to_none=True)
            with deterministic_policy(policy):
                lp,entropy=rollout_stats(policy,gen)
                loss,diag=grpo_policy_loss(lp,old,advantages,mask,ref,float(cfg['clip_epsilon']),float(cfg['kl_beta']),loss_type,int(cfg['max_completion_length']))
                stats=length_gradient_statistics(lp,loss,mask)
                length_stats.extend({'update':step+1,'epoch':epoch,**identity(row),**s} for row,s in zip(expanded,stats))
                loss.backward();norms.append(norm_and_step(policy,optimizer,cfg['max_grad_norm']))
                losses.append(float(diag['policy_term']))
        with deterministic_policy(policy),torch.no_grad():lp,entropy=rollout_stats(policy,gen)
        run.log(update=step+1,reward=float(rewards.mean()),kl=float(sampled_kl(lp,ref,original_mask)),
                entropy=float(masked_mean(entropy,original_mask)),policy_loss=float(np.mean(losses)),gradient_norm=float(np.mean(norms)),
                response_length=float(original_mask.sum(-1).mean()),generated_tokens=int(original_mask.sum()),
                truncated_fraction=float(np.mean(gen['truncated'])),valid_training_tokens=int(mask.sum()),
                ids=[identity(r) for r in selected],**group_statistics(rewards,gids))
        traces.extend({**identity(row),'update':step+1,'response':response,'reward':float(r),'response_tokens':n,'truncated':t}
                      for row,response,r,n,t in zip(expanded,gen['responses'],rewards,gen['response_lengths'],gen['truncated']))
    out.mkdir(parents=True,exist_ok=True);policy.save_pretrained(out);tok.save_pretrained(out)
    jsonl(run.directory/'rollouts.jsonl',traces);jsonl(run.directory/'length_gradients.jsonl',length_stats)
    run.finish(adapter=str(out),generated_tokens=sum(r['generated_tokens'] for r in run.records),
               normalization_denominator='realized valid sequence length' if loss_type=='grpo' else int(cfg['max_completion_length']))
    return str(out)


def evaluate(config,adapter,name):
    cfg=load_yaml(config);set_seed(int(cfg['seed']));rows=read_jsonl(cfg['paths']['rl_prompt_eval'])
    tok=load_tokenizer(cfg['base_model']);policy=load_policy(cfg,adapter,False)
    metrics,records=generate_evaluation(cfg,rows,tok,policy,load_reward_model(cfg),int(cfg['max_completion_length']))
    jsonl(f"{cfg['results_dir']}/{name}/generations.jsonl",records)
    dump(f"{cfg['results_dir']}/{name}/metrics.json",{'config':cfg,'adapter':adapter,**metrics});print(metrics)
    return metrics


def regroup(by_prompt,k):
    if k not in (2,4,8):raise ValueError('Only fixed K=2,4,8 conditions are supported')
    groups=[]
    for pid in sorted(by_prompt):
        # Use the same first eight completions once for every K; never pool different prompts.
        completions=sorted(by_prompt[pid],key=lambda r:int(r['generation_index']))[:8]
        if len(completions)!=8:raise ValueError(f'{pid}: need eight completions')
        if len({r['generation_index'] for r in completions})!=8:raise ValueError(f'{pid}: duplicate cached generation index')
        groups.extend(completions[start:start+k] for start in range(0,8,k))
    return groups


def analyze_groups(by_prompt,sizes=(2,4,8)):
    if not by_prompt:raise ValueError('Empty group cache')
    # Reward model scores are continuous: define difficulty from the SAME K=8 mean reward.
    means={pid:float(np.mean([r['reward'] for r in sorted(group,key=lambda r:int(r['generation_index']))[:8]])) for pid,group in by_prompt.items()}
    threshold=float(np.median(list(means.values())))
    bins={pid:'hard' if value<=threshold else 'easy' for pid,value in means.items()}
    results=[]
    for k in sizes:
        groups=regroup(by_prompt,k)
        for name in ('all','hard','easy'):
            selected=[g for g in groups if name=='all' or bins[str(g[0]['source_index'])]==name]
            if not selected:
                results.append({'k':k,'difficulty':name,'groups':0,'status':'empty_bin'});continue
            rewards=torch.tensor([r['reward'] for g in selected for r in g],dtype=torch.float32)
            gids=torch.arange(len(selected)).repeat_interleave(k)
            results.append({'k':k,'difficulty':name,'generations':len(rewards),**group_statistics(rewards,gids)})
    totals=[r['generations'] for r in results if r['difficulty']=='all']
    if len(set(totals))!=1:raise AssertionError('Generation budget changed across K')
    return {'difficulty_rule':'hard: per-prompt mean reward over fixed first eight completions <= median; easy: greater',
            'difficulty_threshold':threshold,'prompt_bins':bins,'rows':results}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',default='configs/grpo.yaml')
    ap.add_argument('--mode',choices=['standard','groups','normalization','evaluate','all'],default='all')
    ap.add_argument('--adapter');ap.add_argument('--name',default='standard');args=ap.parse_args();cfg=load_yaml(args.config)
    if args.mode not in ('groups','evaluate'):require_accelerator()
    if args.mode in ('standard','all'):
        path=train(args.config);release_memory();evaluate(args.config,path,'standard');release_memory()
    if args.mode in ('groups','all'):
        results=analyze_groups(load_k8_cache(cfg['group_cache']),cfg['group_sizes'])
        dump(f"{cfg['results_dir']}/group_size.json",results);print(results)
    if args.mode in ('normalization','all'):
        for loss_type in ('grpo','dr_grpo'):
            name=f'normalization_{loss_type}';path=train(args.config,name,cfg['fork_updates'],loss_type)
            release_memory();evaluate(args.config,path,name);release_memory()
    if args.mode=='evaluate':evaluate(args.config,args.adapter or cfg['output'],args.name)


if __name__=='__main__':main()
