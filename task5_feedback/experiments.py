"""Matched GSM8K/SVAMP evaluation and controlled reward diagnostics."""
from __future__ import annotations
import argparse
import time
from collections import Counter
import numpy as np

from common.data import load_yaml,read_jsonl,repo_path,prompt_messages
from common.generation import batch_generate
from common.models import load_tokenizer
from common.logging_utils import set_seed
from common.experiment_utils import dump,jsonl,identity,length_summary,release_memory,assert_matching_ids
from task5_feedback.evaluate_math import policy_specs,dataset_path,load_frozen_policy
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward,extract_designated_final
from task5_feedback.score_perturbations import load_diagnostic_groups,EXPECTED_VARIANTS


def field(row,*names):
    for name in names:
        if name in row:return row[name]
    raise ValueError(f'Missing one of {names}; available keys: {sorted(row)}')


def math_metrics(rows):
    if not rows:raise ValueError('No math results')
    return {'n':len(rows),'exact_accuracy':float(np.mean([r['exact_reward'] for r in rows])),
            'format_compliance':float(np.mean([r['format_compliant'] for r in rows])),**length_summary([r['response_tokens'] for r in rows])}


def signal_agreement(a,b,preference):
    verifier='A' if a>b else 'B' if b>a else 'TIE'
    return verifier==preference,verifier


def evaluate_dataset(cfg,dataset):
    started=time.perf_counter()
    directory=repo_path(cfg['results_dir'])/'task5_feedback'/dataset
    rows=read_jsonl(dataset_path(cfg,dataset));tok=load_tokenizer(cfg['base_model']);datasets={}
    for name in policy_specs(cfg):
        target=directory/f'generated_{name}.jsonl'
        if target.exists():
            saved=read_jsonl(target)
            if [identity(r) for r in saved]!=[identity(r) for r in rows]:raise ValueError('Math cache IDs differ')
            import json
            meta=json.loads((directory/'generation_config.json').read_text())
            if meta!=cfg:raise ValueError('Math generation config changed; archive old generations')
            datasets[name]=saved;continue
        set_seed(int(cfg['seed']));model=load_frozen_policy(cfg,name);records=[]
        for row in rows:
            gen=batch_generate(model,tok,[prompt_messages(row)],256,int(cfg['math_max_new_tokens']),do_sample=False)
            response=gen['responses'][0];gold=str(field(row,'gold_final','answer'))
            records.append({**identity(row),'policy':name,'question':str(field(row,'question','prompt')),'gold_final':gold,
                            'response':response,'response_tokens':gen['response_lengths'][0],
                            'exact_reward':exact_reward(response,gold),'format_compliant':extract_designated_final(response) is not None})
        jsonl(target,records);dump(directory/'generation_config.json',cfg);datasets[name]=records
        del model;release_memory()
    # Exact fixed order and IDs are mandatory for every comparison.
    for name,records in datasets.items():
        if [identity(r) for r in records]!=[identity(r) for r in rows]:raise ValueError(f'{name}: mismatched math IDs')
    judge_started=time.perf_counter()
    judge=PairwiseAIJudge(cfg,directory/'judge_cache.json');metrics={};comparisons=[]
    judge_load_seconds=time.perf_counter()-judge_started
    compare_started=time.perf_counter();cached_before=len(judge.cache)
    for name,records in datasets.items():
        summary=math_metrics(records);prefs=[];agreements=[];informative=[]
        for candidate,base in zip(records,datasets['sft']):
            pref=judge.compare(candidate['question'],candidate['response'],base['response'])
            agree,verifier=signal_agreement(candidate['exact_reward'],base['exact_reward'],pref)
            prefs.append(pref);agreements.append(agree)
            if verifier!='TIE':informative.append(agree)
            comparisons.append({**identity(candidate),'policy':name,'judge_preference':pref,'verifier_preference':verifier,'agreement':agree,
                                'question':candidate['question'],'candidate':candidate['response'],'sft':base['response']})
        summary.update(pairwise_win_rate=float(np.mean([{'A':1.,'TIE':.5,'B':0.}[p] for p in prefs])),judge_ties=prefs.count('TIE'),
                       verifier_judge_agreement=float(np.mean(agreements)),
                       verifier_informative_agreement=float(np.mean(informative)) if informative else None,
                       verifier_informative_pairs=len(informative),pairwise_counts=dict(Counter(prefs)))
        metrics[name]=summary
    jsonl(directory/'comparisons.jsonl',comparisons);dump(directory/'metrics.json',metrics)
    dump(directory/'compute.json',{'wall_seconds':time.perf_counter()-started,'judge_load_seconds':judge_load_seconds,
         'pairwise_seconds':time.perf_counter()-compare_started,'new_cached_comparisons':len(judge.cache)-cached_before,
         'total_cached_comparisons':len(judge.cache),'comparison_requests':len(comparisons),
         'tie_parser_note':'The unchanged supplied parser maps unparseable judgments to TIE.'})
    print(dataset,metrics);return metrics


def controlled_pairs(groups,judge):
    results=[]
    verifier_seconds=0.;judge_seconds=0.
    for pid,variants in groups.items():
        clean=variants['clean_correct'];problem=str(field(clean,'question','problem','prompt'));gold=str(field(clean,'gold_final','answer'))
        clean_text=str(field(clean,'response','completion'))
        for kind in sorted(EXPECTED_VARIANTS):
            variant=variants[kind];text=str(field(variant,'response','completion'))
            if str(field(variant,'gold_final','answer'))!=gold:raise ValueError('Gold answer differs within diagnostic group')
            started=time.perf_counter();a=exact_reward(clean_text,gold);b=exact_reward(text,gold)
            verifier_seconds+=time.perf_counter()-started
            _,verifier=signal_agreement(a,b,'TIE')
            started=time.perf_counter();ai=judge.compare(problem,clean_text,text)
            judge_seconds+=time.perf_counter()-started
            results.append({'problem_id':pid,'variant_type':kind,'question':problem,'gold_final':gold,'clean':clean_text,'variant':text,
                            'verifier_preference':verifier,'ai_preference':ai,'verifier_clean_reward':a,'verifier_variant_reward':b})
    summary=[]
    for kind in sorted(EXPECTED_VARIANTS):
        subset=[r for r in results if r['variant_type']==kind]
        for mechanism,key in [('rlvr','verifier_preference'),('rlaif','ai_preference')]:
            prefs=[r[key] for r in subset]
            summary.append({'variant_type':kind,'mechanism':mechanism,'n':len(subset),
                            'comparison':'identical clean control' if kind=='clean_correct' else 'clean versus perturbed',
                            'better_rate':prefs.count('A')/len(prefs),'tie_rate':prefs.count('TIE')/len(prefs),'wrong_rate':prefs.count('B')/len(prefs)})
    sensitivity={}
    for mechanism,key in [('rlvr','verifier_preference'),('rlaif','ai_preference')]:
        reasoning=[r for r in results if r['variant_type']=='corrupt_reasoning_correct_final']
        outcome=[r for r in results if r['variant_type']=='good_reasoning_wrong_final']
        sensitivity[mechanism]={'S_reason':float(np.mean([r[key]=='A' for r in reasoning])),
                                'S_outcome':float(np.mean([r[key]=='A' for r in outcome]))}
    return {'rows':summary,'sensitivity':sensitivity,'verifier_seconds':verifier_seconds,'judge_compare_seconds':judge_seconds,
            'note':'clean_correct is a self-comparison tie control; a verifier tie on reasoning-only changes is an invariance, not an accuracy error'},results


def diagnostics(cfg):
    groups=load_diagnostic_groups(cfg['paths']['task5_diagnostics'])
    if len(groups)!=20 or any(len(g)!=5 for g in groups.values()):raise ValueError('Expected fixed 20 x 5 diagnostic set')
    directory=repo_path(cfg['results_dir'])/'task5_feedback'/'diagnostics'
    judge=PairwiseAIJudge(cfg,directory/'judge_cache.json');summary,pairs=controlled_pairs(groups,judge)
    dump(directory/'metrics.json',summary);jsonl(directory/'pairs.jsonl',pairs);dump(directory/'config.json',cfg);print(summary)
    return summary


def compare(cfg):
    import json
    directory=repo_path(cfg['results_dir'])/'task5_feedback'
    gsm=json.loads((directory/'gsm'/'metrics.json').read_text());transfer=json.loads((directory/'transfer'/'metrics.json').read_text())
    comparisons=[]
    for policy in policy_specs(cfg):
        a,b=gsm[policy],transfer[policy]
        comparisons.append({'policy':policy,'gsm':a,'transfer':b,
                            'exact_accuracy_drop':a['exact_accuracy']-b['exact_accuracy'],
                            'pairwise_win_rate_drop':a['pairwise_win_rate']-b['pairwise_win_rate']})
    failures=[]
    for dataset in ('gsm','transfer'):
        for policy in policy_specs(cfg):
            rows=read_jsonl(directory/dataset/f'generated_{policy}.jsonl')
            counts=Counter('format_failure' if not r['format_compliant'] else 'wrong_final' if not r['exact_reward'] else 'correct' for r in rows)
            failures.append({'dataset':dataset,'policy':policy,'counts':dict(counts)})
    dump(directory/'comparison.json',{'rows':comparisons,'failure_types':failures});print(comparisons)
    return comparisons


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',default='configs/feedback.yaml')
    ap.add_argument('--mode',choices=['gsm','transfer','diagnostics','compare','all'],default='all');args=ap.parse_args();cfg=load_yaml(args.config)
    if args.mode in ('gsm','all'):evaluate_dataset(cfg,'gsm');release_memory()
    if args.mode in ('diagnostics','all'):diagnostics(cfg);release_memory()
    if args.mode in ('transfer','all'):evaluate_dataset(cfg,'transfer');release_memory()
    if args.mode in ('compare','all'):compare(cfg)


if __name__=='__main__':main()
