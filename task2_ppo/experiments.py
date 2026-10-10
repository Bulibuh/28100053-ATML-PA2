"""PPO continuation, cached clipping diagnostics, and matched forks."""
from __future__ import annotations
import argparse
import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path, prompt_messages
from common.generation import batch_generate, score_reward_pairs
from common.models import load_policy, load_tokenizer, load_reward_model, load_value_model
from common.logging_utils import set_seed
from common.metrics import masked_mean, sampled_kl
from common.experiment_utils import (Run, dump, jsonl, deterministic_policy, rollout_stats, response_values, norm_and_step,
    generate_evaluation, require_accelerator, release_memory, identity, trainable_rollout)
from task2_ppo.continue_train import prepare_ppo_continuation
from task2_ppo.analyze_clipping import load_cached_rollouts
from task2_ppo.ppo import compute_gae, shaped_rewards, normalize_advantages, ppo_policy_loss, value_mse_loss


def train(config='configs/ppo.yaml', name='standard', updates=None, clip_epsilon=None, kl_beta=None):
    b=prepare_ppo_continuation(config); cfg=dict(b['cfg'])
    for key,value in [('updates',updates),('clip_epsilon',clip_epsilon),('kl_beta',kl_beta)]:
        if value is not None:cfg[key]=value
    cfg['policy_dropout_enabled_during_ratio_updates']=False
    cfg['allocated_response_token_budget']=int(cfg['updates'])*int(cfg['prompts_per_update'])*int(cfg['max_response_length'])
    policy,value=b['policy'],b['value_model']; po,vo=b['policy_optimizer'],b['value_optimizer']
    rows=b['prompt_rows']; tok=b['tokenizer']; out=repo_path(f'outputs/task2_ppo/{name}')
    if not rows:raise ValueError('Empty PPO prompt pool')
    if (out/'adapter_config.json').exists():raise FileExistsError(out)
    run=Run(cfg['results_dir'],name,cfg,rows); traces=[]
    for step in range(int(cfg['updates'])):
        selected=[rows[(step*int(cfg['prompts_per_update'])+j)%len(rows)] for j in range(int(cfg['prompts_per_update']))]
        prompts=[prompt_messages(r) for r in selected]
        gen=batch_generate(policy,tok,prompts,int(cfg['max_prompt_length']),int(cfg['max_response_length']),**cfg['generation'])
        gen=trainable_rollout(gen)
        mask=gen['response_mask']
        with deterministic_policy(policy),deterministic_policy(value),torch.no_grad():
            old,_=rollout_stats(policy,gen); ref,_=rollout_stats(policy,gen,True); old_values=response_values(value,gen)
            task=score_reward_pairs(b['reward_model'],b['reward_tokenizer'],prompts,gen['responses'],int(cfg['reward_max_length']))
            penalized=task-torch.tensor([not t for t in gen['terminated_with_eos']],device=task.device)*float(cfg['missing_eos_penalty'])
            rewards=shaped_rewards(penalized,old,ref,mask,cfg['kl_beta'])
            advantages,returns=compute_gae(rewards,old_values,mask,float(cfg['gamma']),float(cfg['gae_lambda']))
            advantages=normalize_advantages(advantages,mask).detach();returns=returns.detach()
        epoch_stats=[]
        for _ in range(int(cfg['ppo_epochs'])):
            po.zero_grad(set_to_none=True);vo.zero_grad(set_to_none=True)
            with deterministic_policy(policy),deterministic_policy(value):
                lp,entropy=rollout_stats(policy,gen)
                pl,ratio,clip=ppo_policy_loss(lp,old,advantages,mask,float(cfg['clip_epsilon']))
                pv=response_values(value,gen);vl=value_mse_loss(pv,returns,mask)
                pl.backward();(float(cfg['value_coef'])*vl).backward()
                pn=norm_and_step(policy,po,cfg['max_grad_norm']);vn=norm_and_step(value,vo,cfg['max_grad_norm'])
            epoch_stats.append({'policy_loss':float(pl.detach()),'value_loss':float(vl.detach()),'clip_fraction':float(clip),
                                'gradient_norm':pn,'value_gradient_norm':vn})
        with deterministic_policy(policy),torch.no_grad():
            lp,entropy=rollout_stats(policy,gen)
        log={k:float(np.mean([r[k] for r in epoch_stats])) for k in epoch_stats[0]}
        log.update(update=step+1,reward=float(task.mean()),penalized_reward=float(penalized.mean()),kl=float(sampled_kl(lp,ref,mask)),
                   entropy=float(masked_mean(entropy,mask)),response_length=float(mask.sum(-1).mean()),
                   generated_tokens=int(mask.sum()),ids=[identity(r) for r in selected],
                   value_return_mse=float(value_mse_loss(old_values,returns,mask)))
        run.log(**log)
        traces.extend({**identity(r),'update':step+1,'response':resp,'reward':float(rew),'response_tokens':length} for r,resp,rew,length in zip(selected,gen['responses'],task,gen['response_lengths']))
    out.mkdir(parents=True,exist_ok=True);policy.save_pretrained(out);tok.save_pretrained(out)
    value.save_pretrained(out/'value');jsonl(run.directory/'rollouts.jsonl',traces)
    run.finish(adapter=str(out),stability_statistic='std of pre-clipping policy gradient norm across updates',
               gradient_norm_std=float(np.std([r['gradient_norm'] for r in run.records])),
               generated_tokens=sum(r['generated_tokens'] for r in run.records))
    return str(out)


def evaluate(config,adapter,name):
    cfg=load_yaml(config);set_seed(int(cfg['seed']))
    rows=read_jsonl(cfg['paths']['rl_prompt_eval']);tok=load_tokenizer(cfg['base_model']);policy=load_policy(cfg,adapter,False)
    metrics,records=generate_evaluation(cfg,rows,tok,policy,load_reward_model(cfg),int(cfg['eval_max_response_length']))
    jsonl(f"{cfg['results_dir']}/{name}/generations.jsonl",records)
    dump(f"{cfg['results_dir']}/{name}/metrics.json",{'config':cfg,'adapter':adapter,**metrics});print(metrics)
    return metrics


def clipping_statistics(new,old,advantage,mask,eps):
    loss,_,affected=ppo_policy_loss(new,old,advantage,mask,eps)
    return {'epsilon':eps,'clipped_surrogate':-float(loss),'clip_fraction':float(affected),'affected_token_fraction':float(affected),'valid_tokens':int(mask.sum())}


def cached_generation(row,tok,device):
    """Recover saved token IDs where available; otherwise verify reconstructed length exactly."""
    old=torch.as_tensor(row['old_logprobs'],dtype=torch.float32,device=device).reshape(1,-1)
    if 'response_ids' in row:
        response=torch.as_tensor(row['response_ids'],dtype=torch.long,device=device).reshape(1,-1)
    else:
        ids=tok(row['response'],add_special_tokens=False)['input_ids']
        if len(ids)+1==old.shape[1]:ids=ids+[tok.eos_token_id]
        if len(ids)!=old.shape[1]:raise ValueError('Cache response/token mismatch; exact response_ids are required, refusing to silently truncate cached log-probs')
        response=torch.tensor([ids],device=device)
    if 'sequences' in row or 'input_ids' in row:
        sequence=torch.as_tensor(row.get('sequences',row.get('input_ids')),device=device).reshape(1,-1)
        width=sequence.shape[1]-response.shape[1]
    else:
        messages = prompt_messages(row)

        if not any(m.get("content") for m in messages):
            eval_rows = read_jsonl("data/rl_prompt_pool_eval.jsonl")
    
            matches = [
                p for p in eval_rows
                if str(p.get("prompt_id")) == str(row.get("prompt_id"))
                and str(p.get("source_index")) == str(row.get("source_index"))
            ]
    
            if len(matches) != 1:
                raise ValueError("Could not uniquely match cached prompt")
    
            messages = prompt_messages(matches[0])
    
        prompt = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        sequence=torch.cat([torch.tensor([prompt],device=device),response],dim=1);width=len(prompt)
    if width<1:raise ValueError('Missing cached prompt prefix')
    mask=torch.as_tensor(row.get('response_mask',torch.ones_like(old)),device=device,dtype=torch.float32).reshape(1,-1)
    if old.shape!=response.shape or mask.shape!=old.shape:raise ValueError('Cached token/mask dimensions differ')
    attention=torch.as_tensor(row.get('attention_mask',torch.ones_like(sequence)),device=device).reshape(1,-1)
    return {'sequences':sequence,'response_ids':response,'attention_mask':attention,'response_mask':mask,'prompt_width':width},old


def analyze_cache(config):
    cfg=load_yaml(config);set_seed(int(cfg['seed']));rows=load_cached_rollouts(cfg['cached_rollouts'])
    tok=load_tokenizer(cfg['base_model']);policy=load_policy(cfg,cfg['paths']['ppo_midpoint_policy'],False)
    value=load_value_model(cfg,cfg['paths']['ppo_midpoint_value'],train_mode='frozen');device=next(policy.parameters()).device
    blocks=[]
    with torch.no_grad():
        for row in rows:
            gen,old=cached_generation(row,tok,device);mask=gen['response_mask']
            ref=torch.as_tensor(row['ref_logprobs'],device=device,dtype=torch.float32).reshape_as(old)
            values=torch.as_tensor(row['values'],device=device,dtype=torch.float32).reshape_as(old) if 'values' in row else response_values(value,gen)
            reward=next((row[k] for k in ('effective_terminal_reward','task_reward','reward','reward_score','rm_reward') if k in row),None)
            if reward is None:raise ValueError('Cached rollout has no learned task reward')
            reward=torch.as_tensor(reward,device=device,dtype=torch.float32).reshape(1)
            if 'effective_terminal_reward' not in row and not row.get('terminated_with_eos',True):
                reward=reward-float(cfg['missing_eos_penalty'])
            shaped=shaped_rewards(reward,old,ref,mask,float(cfg['kl_beta']))
            adv,_=compute_gae(shaped,values,mask,float(cfg['gamma']),float(cfg['gae_lambda']))
            new,_=rollout_stats(policy,gen)
            blocks.append((new.flatten(),old.flatten(),adv.flatten(),mask.flatten()))
    new,old,adv,mask=[torch.cat([b[i] for b in blocks]).unsqueeze(0) for i in range(4)]
    adv=normalize_advantages(adv,mask)
    stats=[clipping_statistics(new,old,adv,mask,float(e)) for e in cfg['clip_values']]
    dump(f"{cfg['results_dir']}/cached_clipping.json",{'config':cfg,'policy_checkpoint':cfg['paths']['ppo_midpoint_policy'],'ids':[identity(r) for r in rows],'rows':stats})
    print(stats);return stats


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',default='configs/ppo.yaml')
    ap.add_argument('--mode',choices=['standard','clipping','cache','kl','evaluate','all'],default='all');ap.add_argument('--adapter');ap.add_argument('--name',default='standard')
    args=ap.parse_args();cfg=load_yaml(args.config)
    if args.mode not in ('cache','evaluate'):require_accelerator()
    if args.mode in ('standard','all'):
        path=train(args.config);release_memory();evaluate(args.config,path,'standard');release_memory()
    if args.mode in ('cache','clipping','all'):analyze_cache(args.config);release_memory()
    if args.mode in ('clipping','all'):
        for eps in cfg['clip_values']:
            name=f'clip_{eps:.2f}';path=train(args.config,name,cfg['fork_updates'],clip_epsilon=eps);release_memory();evaluate(args.config,path,name);release_memory()
    if args.mode in ('kl','all'):
        for beta in cfg['kl_values']:
            name=f'kl_{beta:.2f}';path=train(args.config,name,cfg['fork_updates'],kl_beta=beta);release_memory();evaluate(args.config,path,name);release_memory()
    if args.mode=='evaluate':evaluate(args.config,args.adapter or cfg['output'],args.name)


if __name__=='__main__':main()
