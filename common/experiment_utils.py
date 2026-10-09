"""Reproducible experiment bookkeeping and shared rollout/evaluation primitives."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import platform
import time
from contextlib import contextmanager
from pathlib import Path
from importlib.metadata import version

import numpy as np
import torch

from common.data import repo_path, prompt_messages
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.metrics import masked_mean, sampled_kl
from common.models import reference_mode, token_values, trainable_parameters


def identity(row):
    keys = ('source_split', 'source_index', 'prompt_id', 'xstest_id', 'problem_id', 'generation_index')
    found = {k: row[k] for k in keys if k in row}
    if not found:
        raise ValueError('Fixed data row has no source/prompt ID')
    return found


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, torch.Tensor):
        return clean_json(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump(path, obj):
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(obj), indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def jsonl(path, rows):
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(clean_json(r), ensure_ascii=False, allow_nan=False)+'\n' for r in rows))


def require_accelerator():
    if not torch.cuda.is_available():
        raise RuntimeError('Full course training requires a CUDA GPU. Run pytest and the CPU notebook diagnostics here; run the full commands on a CUDA machine with course assets.')


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def deterministic_policy(model):
    """Disable LoRA dropout without disabling gradients; old/new ratios use one policy."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def rollout_stats(policy, gen, reference=False):
    if reference:
        with torch.no_grad(), reference_mode(policy):
            lp, _ = response_token_logprobs(policy, gen['sequences'], gen['attention_mask'], gen['prompt_width'], gen['response_ids'])
        return lp.detach(), None
    lp, logits = response_token_logprobs(policy, gen['sequences'], gen['attention_mask'], gen['prompt_width'], gen['response_ids'])
    with torch.no_grad():
        logdist = torch.log_softmax(logits.detach().float(), -1)
        entropy = -(logdist.exp()*logdist).sum(-1)
    return lp, entropy


def trainable_rollout(gen):
    """Copy tensors made by the released inference-mode generator for autograd updates."""
    return {key:value.clone() if isinstance(value,torch.Tensor) else value for key,value in gen.items()}


def response_values(model, gen):
    # V(s_t) is the hidden state BEFORE action token t, matching log-prob indexing.
    v = token_values(model, gen['sequences'], gen['attention_mask'])
    return v[:, gen['prompt_width']-1:-1][:, :gen['response_ids'].shape[1]].float()


def norm_and_step(model, optimizer, max_norm):
    params = trainable_parameters(model)
    norm = torch.nn.utils.clip_grad_norm_(params, float(max_norm))
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite gradient norm; refusing to save a corrupt checkpoint')
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


class Run:
    def __init__(self, directory, name, cfg, rows):
        self.directory = repo_path(directory)/name
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory/'run.json').exists():
            raise FileExistsError(f'Run already exists: {self.directory}. Choose another name or archive the previous run.')
        self.start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        packages = {}
        for name_ in ('torch', 'transformers', 'peft', 'trl', 'tokenizers'):
            try: packages[name_] = version(name_)
            except Exception: pass
        self.metadata = {'config': cfg, 'ids': [identity(r) for r in rows], 'python':platform.python_version(), 'packages':packages,
                         'cuda':torch.cuda.is_available(), 'kl_convention':'response-token masked mean of log policy minus log frozen base reference',
                         'entropy_convention':'categorical entropy, response-token masked mean', 'completed':False}
        root=repo_path('.')
        files=[p for directory in ('common','scripts','task1_dpo','task2_ppo','task3_grpo','task4_safety','task5_feedback','task6_synthesis','configs')
               for p in (root/directory).glob('*') if p.is_file() and p.suffix in ('.py','.yaml')]
        self.metadata['code_sha256']={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
        dump(self.directory/'run.json', self.metadata)
        self.records = []

    def log(self, **record):
        self.records.append(clean_json(record))
        jsonl(self.directory/'training.jsonl', self.records)
        print(json.dumps(clean_json(record), ensure_ascii=False))

    def finish(self, **extra):
        self.metadata.update(extra, completed=True, wall_seconds=time.perf_counter()-self.start,
                             peak_vram_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
        dump(self.directory/'run.json', self.metadata)


def length_summary(lengths):
    return {'response_length_mean':float(np.mean(lengths)), 'response_length_std':float(np.std(lengths)),
            'response_length_q25':float(np.quantile(lengths,.25)), 'response_length_q75':float(np.quantile(lengths,.75))}


def generate_evaluation(cfg, rows, tokenizer, policy, reward, cap):
    records=[]
    rm,rt = reward
    with deterministic_policy(policy), torch.no_grad():
        for row in rows:
            prompts=[prompt_messages(row)]
            gen=batch_generate(policy, tokenizer, prompts, int(cfg.get('max_prompt_length',256)), cap, do_sample=False)
            lp,entropy=rollout_stats(policy,gen)
            ref,_=rollout_stats(policy,gen,reference=True)
            mask=gen['response_mask']
            score=score_reward_pairs(rm,rt,prompts,gen['responses'],int(cfg.get('reward_max_length',1280)))
            records.append({**identity(row), 'prompt':prompts, 'response':gen['responses'][0], 'response_tokens':gen['response_lengths'][0],
                            'reward':float(score[0]), 'kl_sum':float(((lp-ref)*mask).sum()),
                            'entropy_sum':float((entropy*mask).sum()), 'terminated':gen['terminated_with_eos'][0]})
    tokens=sum(r['response_tokens'] for r in records)
    if not tokens:raise ValueError('No evaluation response tokens')
    metrics={'n':len(records),'reward_mean':float(np.mean([r['reward'] for r in records])),
             'kl':sum(r['kl_sum'] for r in records)/tokens,'entropy':sum(r['entropy_sum'] for r in records)/tokens,
             **length_summary([r['response_tokens'] for r in records])}
    return metrics,records


def assert_matching_ids(*datasets, key):
    expected=[r[key] for r in datasets[0]]
    if len(expected)!=len(set(expected)):
        raise ValueError('Duplicate evaluation IDs')
    for rows in datasets[1:]:
        if [r[key] for r in rows]!=expected:raise ValueError('Evaluation IDs/order differ between conditions')
