"""Validate downloaded fixed data without modifying it; skip only when assets are absent."""
from pathlib import Path
from collections import Counter
import pytest
import pandas as pd
from common.data import REPO_ROOT,read_jsonl
from task5_feedback.rlvr import exact_reward
from task4_safety.make_audit_sheet import fixed_audit_ids


def needed(path):
    p=REPO_ROOT/path
    if not p.exists():pytest.skip(f'Course asset is not installed: {path}')
    return p


def test_fixed_diagnostic_answers_match_all_100_staff_expectations():
    rows=read_jsonl(needed('data/task5_controlled_reward_diagnostics.jsonl'))
    assert len(rows)==100 and len({r['problem_id'] for r in rows})==20
    assert all(r['manual_validation'] for r in rows)
    assert all(exact_reward(r['response'],str(r['gold_final']))==r['expected_exact_reward'] for r in rows)
    assert set(Counter(r['variant_type'] for r in rows).values())=={20}


def test_fixed_length_strata_and_audit_ids():
    train=read_jsonl(needed('data/dpo_length_balanced_train.jsonl'))
    evaluation=read_jsonl(needed('data/dpo_length_stratified_eval.jsonl'))
    assert set(Counter(r['length_stratum'] for r in train).values())=={500}
    assert set(Counter(r['length_stratum'] for r in evaluation).values())=={82}
    assert set((r['source_split'],r['source_index']) for r in train).isdisjoint((r['source_split'],r['source_index']) for r in evaluation)
    prompts=pd.read_csv(needed('data/xstest_safety_prompts.csv'))
    ids=fixed_audit_ids(prompts.to_dict('records'),30,6304)
    assert len(ids)==len(set(ids))==60
    assert prompts[prompts['xstest_id'].isin(ids)]['benchmark_class'].value_counts().to_dict()=={'SAFE':30,'UNSAFE':30}
