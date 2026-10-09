"""Collect evidence availability, metrics and compute from Tasks 1-5."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from common.data import repo_path
from common.experiment_utils import dump
from common.visualization import flat_metrics

REQUIRED={
    'task1_dpo':['standard','beta_0.03','beta_0.10','beta_0.30','standard_length','balanced_length'],
    'task2_ppo':['standard','clip_0.05','clip_0.20','clip_0.50','kl_0.00','kl_0.10','kl_0.20'],
    'task3_grpo':['standard','normalization_grpo','normalization_dr_grpo'],
    'task4_safety':['.'],
    'task5_feedback':['gsm','transfer','diagnostics'],
}


def summarize(results='results',save=True):
    root=repo_path(results);evidence=[];compute=[]
    for task,conditions in REQUIRED.items():
        for condition in conditions:
            directory=root/task/condition;path=directory/'metrics.json'
            if path.exists():
                metrics=json.loads(path.read_text())
                if task in ('task4_safety','task5_feedback') and condition!='diagnostics':
                    for policy,value in metrics.items():evidence.append({'task':task,'condition':condition,'policy':policy,'status':'saved',**flat_metrics(value)})
                else:evidence.append({'task':task,'condition':condition,'status':'saved',**flat_metrics(metrics)})
            else:evidence.append({'task':task,'condition':condition,'status':'not run'})
            if (directory/'run.json').exists():
                run=json.loads((directory/'run.json').read_text());compute.append({'task':task,'condition':condition,**{k:run.get(k) for k in ('completed','wall_seconds','peak_vram_bytes','generated_tokens','examples_seen')}})
    for task,filename in [('task2_ppo','cached_clipping.json'),('task3_grpo','group_size.json'),('task5_feedback','comparison.json'),('task4_safety','audit_metrics.json')]:
        path=root/task/filename
        status='not run'
        if path.exists():
            value=json.loads(path.read_text());status='manual audit incomplete' if filename=='audit_metrics.json' and not value.get('complete') else 'saved'
        evidence.append({'task':task,'condition':filename,'status':status})
    if save:
        dump(root/'task6_synthesis'/'evidence.json',{'rows':evidence,'compute':compute})
        pd.DataFrame(evidence).to_csv(root/'task6_synthesis'/'evidence.csv',index=False)
    return pd.DataFrame(evidence),pd.DataFrame(compute)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',default='results');args=ap.parse_args()
    evidence,compute=summarize(args.results);print(evidence.to_string(index=False));print(compute.to_string(index=False))


if __name__=='__main__':main()
