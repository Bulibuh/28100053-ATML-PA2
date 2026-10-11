"""Frozen-policy safety evaluation and a blind, human-completed audit."""
from __future__ import annotations
import argparse
import hashlib
from collections import Counter
import numpy as np
import pandas as pd

from common.data import load_yaml,read_jsonl,repo_path
from common.logging_utils import set_seed
from common.experiment_utils import dump,jsonl,length_summary,release_memory,assert_matching_ids
from task4_safety.generate_responses import generate_for_policy,policy_specs
from task4_safety.judge_responses import LABELS,load_judge,judge_one
from task4_safety.make_audit_sheet import fixed_audit_ids


def aggregate(rows):
    if not rows:raise ValueError('No safety results')
    if any(r['label'] not in LABELS for r in rows):raise ValueError('Unknown judge label')
    safe=[r for r in rows if r['benchmark_class']=='SAFE'];unsafe=[r for r in rows if r['benchmark_class']=='UNSAFE']
    def rate(subset,label):return sum(r['label']==label for r in subset)/len(subset) if subset else None
    return {'n':len(rows),'safe_n':len(safe),'unsafe_n':len(unsafe),'safe_answer_rate':rate(safe,'SAFE_ANSWER'),
            'safe_over_refusal_rate':rate(safe,'OVER_REFUSAL'),'unsafe_compliance_rate':rate(unsafe,'UNSAFE_COMPLIANCE'),
            'justified_refusal_rate':rate(unsafe,'JUSTIFIED_REFUSAL'),'ambiguous_rate':rate(rows,'AMBIGUOUS'),
            'label_counts':dict(Counter(r['label'] for r in rows)),**length_summary([r['response_tokens'] for r in rows])}


def outdir(cfg):return repo_path(cfg['results_dir'])/'task4_safety'


def generate(cfg):
    directory=outdir(cfg)
    for policy in policy_specs(cfg):
        path=directory/f'generated_{policy}.jsonl'
        if path.exists():raise FileExistsError(f'{path}: archive existing generations to rerun')
        set_seed(int(cfg['seed']));rows=generate_for_policy(cfg,policy,batch_size=16)
        jsonl(path,rows);release_memory()
    dump(directory/'generation_config.json',cfg)


def judge(cfg):
    directory=outdir(cfg);tok,model=load_judge(cfg)
    for policy in policy_specs(cfg):
        rows=read_jsonl(directory/f'generated_{policy}.jsonl');target=directory/f'judged_{policy}.jsonl'
        saved=read_jsonl(target) if target.exists() else []
        # Resume only an exact prefix of the same generated responses.
        for a,b in zip(saved,rows):
            if any(a[k]!=b[k] for k in ('xstest_id','response','prompt','policy')):raise ValueError('Stale safety judge cache')
        if len(saved)>len(rows):raise ValueError('Safety judge cache longer than input')
        for row in rows[len(saved):]:
            score=judge_one(tok,model,row['prompt'],row['response'],int(cfg['judge_max_new_tokens']))
            saved.append({**row,**score});jsonl(target,saved)
    dump(directory/'judge_config.json',{'model':cfg['ai_judge_model'],'max_new_tokens':cfg['judge_max_new_tokens'],'seed':cfg['seed']})


def make_audit(cfg):
    directory=outdir(cfg);target=directory/'manual_audit.csv'
    if target.exists():raise FileExistsError(f'Preserving manual annotations: {target}')
    datasets={p:read_jsonl(directory/f'generated_{p}.jsonl') for p in policy_specs(cfg)}
    assert_matching_ids(*datasets.values(),key='xstest_id')
    ids=fixed_audit_ids(datasets['sft'],int(cfg['manual_audit_per_class']),int(cfg['seed']))
    lookup=[];sheet=[]
    for policy,rows in datasets.items():
        for row in rows:
            if row['xstest_id'] not in ids:continue
            key=hashlib.sha256(f"{cfg['seed']}:{policy}:{row['xstest_id']}".encode()).hexdigest()[:20]
            lookup.append({'audit_key':key,'policy':policy,'xstest_id':row['xstest_id']})
            sheet.append({'audit_key':key,'prompt':row['prompt'],'response':row['response'],
                          'benchmark_class':row['benchmark_class'],'category':row['type'],'manual_label':''})
    # Policy identities and AI labels are absent from the annotation sheet.
    pd.DataFrame(sheet).sample(frac=1,random_state=int(cfg['seed'])).to_csv(target,index=False)
    dump(directory/'audit_lookup.json',{'ids':ids,'rows':lookup,'seed':cfg['seed']})
    print(f'Fill manual_label in {target} without opening judged files or audit_lookup.json. {len(ids)} IDs x {len(datasets)} policies.')


def audit_agreement(judged,lookup,annotations):
    expected={r['audit_key']:r for r in lookup['rows']}
    if annotations['audit_key'].duplicated().any():raise ValueError('Duplicate manual audit key')
    if set(annotations['audit_key'])!=set(expected):raise ValueError('Manual audit IDs differ from the fixed sheet')
    labeled=[]
    by_key={(r['policy'],r['xstest_id']):r for r in judged}
    for row in annotations.to_dict('records'):
        manual=str(row.get('manual_label','')).strip().upper()
        if not manual or manual=='NAN':continue
        if manual not in LABELS:raise ValueError(f'Unknown manual label: {manual}')
        key=expected[row['audit_key']];ai=by_key[(key['policy'],key['xstest_id'])]
        labeled.append({**key,'manual_label':manual,'ai_label':ai['label'],'prompt':ai['prompt'],'response':ai['response']})
    summary={'expected':len(expected),'labeled':len(labeled),'complete':len(labeled)==len(expected),'policies':{}}
    for policy in sorted({r['policy'] for r in lookup['rows']}):
        subset=[r for r in labeled if r['policy']==policy]
        summary['policies'][policy]={'n':len(subset),'agreement':float(np.mean([r['manual_label']==r['ai_label'] for r in subset])) if subset else None,
            'manual_ambiguous_rate':float(np.mean([r['manual_label']=='AMBIGUOUS' for r in subset])) if subset else None,
            'ai_ambiguous_rate':float(np.mean([r['ai_label']=='AMBIGUOUS' for r in subset])) if subset else None,
            'confusion':{label:dict(Counter(r['ai_label'] for r in subset if r['manual_label']==label)) for label in sorted(LABELS)}}
    return summary,labeled


def summarize(cfg):
    directory=outdir(cfg);datasets={p:read_jsonl(directory/f'judged_{p}.jsonl') for p in policy_specs(cfg)}
    assert_matching_ids(*datasets.values(),key='xstest_id')
    metrics={p:aggregate(rows) for p,rows in datasets.items()};categories=[];examples=[]
    for policy,rows in datasets.items():
        for category in sorted({r['type'] for r in rows}):
            categories.append({'policy':policy,'category':category,**aggregate([r for r in rows if r['type']==category])})
        for label in ('UNSAFE_COMPLIANCE','JUSTIFIED_REFUSAL','OVER_REFUSAL'):
            examples.extend([r for r in rows if r['label']==label][:2])
    dump(directory/'metrics.json',metrics);dump(directory/'categories.json',categories);jsonl(directory/'examples.jsonl',examples)
    audit_path=directory/'manual_audit.csv'
    if audit_path.exists():
        import json
        lookup=json.loads((directory/'audit_lookup.json').read_text())
        audit,labeled=audit_agreement([r for rows in datasets.values() for r in rows],lookup,pd.read_csv(audit_path,keep_default_na=False))
        dump(directory/'audit_metrics.json',audit);jsonl(directory/'audit_labeled.jsonl',labeled)
        print('Manual audit:',audit['labeled'],'/',audit['expected'],'completed; human labels are required for full evidence.')
    else:print('Manual audit not yet created. Run --mode audit and label the sheet.')
    print(metrics);return metrics


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',default='configs/feedback.yaml')
    ap.add_argument('--mode',choices=['generate','judge','audit','summarize','all'],default='all');args=ap.parse_args();cfg=load_yaml(args.config)
    if args.mode in ('generate','all'):generate(cfg)
    if args.mode in ('judge','all'):judge(cfg);release_memory()
    if args.mode in ('audit','all'):make_audit(cfg)
    if args.mode in ('summarize','all'):summarize(cfg)


if __name__=='__main__':main()
