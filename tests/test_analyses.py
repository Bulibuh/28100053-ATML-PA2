import numpy as np
import pandas as pd
import pytest
from task3_grpo.experiments import regroup,analyze_groups
from task4_safety.experiments import aggregate,audit_agreement
from task5_feedback.experiments import controlled_pairs,signal_agreement
from task5_feedback.rlvr import exact_reward,extract_designated_final


def cache():
    return {str(p):[{'source_index':str(p),'generation_index':j,'reward':float(p)+j%2} for j in range(8)] for p in range(4)}


def test_equal_generation_budget_keeps_prompt_groups_separate():
    by_prompt=cache()
    for k in (2,4,8):
        groups=regroup(by_prompt,k)
        assert sum(map(len,groups))==32
        assert all(len({r['source_index'] for r in group})==1 for group in groups)
        assert sorted((r['source_index'],r['generation_index']) for group in groups for r in group)==sorted((r['source_index'],r['generation_index']) for rows in by_prompt.values() for r in rows)
    rows=analyze_groups(by_prompt)['rows']
    assert len(rows)==9
    assert all(r['informative_fraction']==1 for r in rows)


def safety_rows():
    specs=[('SAFE','SAFE_ANSWER'),('SAFE','OVER_REFUSAL'),('SAFE','AMBIGUOUS'),('UNSAFE','UNSAFE_COMPLIANCE'),('UNSAFE','JUSTIFIED_REFUSAL')]
    return [{'xstest_id':i,'policy':'sft','benchmark_class':cls,'label':label,'response_tokens':10,'prompt':'fixture','response':'fixture','type':'test'} for i,(cls,label) in enumerate(specs)]


def test_safety_uses_class_specific_denominators():
    metrics=aggregate(safety_rows())
    assert metrics['safe_answer_rate']==pytest.approx(1/3)
    assert metrics['unsafe_compliance_rate']==.5
    assert metrics['ambiguous_rate']==.2
    assert metrics['response_length_std']==0


def test_audit_missing_labels_remain_missing_and_disagreements_counted():
    rows=safety_rows();lookup={'rows':[{'audit_key':str(i),'policy':'sft','xstest_id':i} for i in range(5)]}
    annotations=pd.DataFrame({'audit_key':[str(i) for i in range(5)],'manual_label':['SAFE_ANSWER','SAFE_ANSWER','','UNSAFE_COMPLIANCE','JUSTIFIED_REFUSAL']})
    result,joined=audit_agreement(rows,lookup,annotations)
    assert result['complete'] is False and result['labeled']==4
    assert result['policies']['sft']['agreement']==.75
    assert result['policies']['sft']['confusion']['SAFE_ANSWER']['OVER_REFUSAL']==1
    assert len(joined)==4


def test_verifier_designated_final_and_distractors():
    assert exact_reward('Gold might be 42, but final:\n#### 7','42')==0
    assert exact_reward('#### 42\nCorrection: #### 7','42')==0
    assert exact_reward('Number 42 appears without a designated field','42')==0
    assert exact_reward('#### 1,000','1000')==1
    assert extract_designated_final('#### -3.5')=='-3.5'


def diagnostic_fixture():
    texts={'clean_correct':'2 + 2 = 4. #### 4','corrupt_reasoning_correct_final':'2 + 2 = 9. #### 4',
           'good_reasoning_wrong_final':'2 + 2 = 4. #### 9','persuasive_filler_correct':'2 + 2 = 4. Believe me. #### 4',
           'gold_distractor_wrong_final':'The candidate is 4, but I choose #### 9'}
    return {'test':{kind:{'question':'2 + 2?','gold_final':'4','response':response} for kind,response in texts.items()}}


class TestJudge:
    """A deterministic test double, never a course AI evaluation."""
    def compare(self,problem,a,b):return 'TIE' if a==b else 'A'


def test_diagnostic_reasoning_invariance_outcome_sensitivity():
    result,pairs=controlled_pairs(diagnostic_fixture(),TestJudge())
    assert result['sensitivity']['rlvr']=={'S_reason':0.,'S_outcome':1.}
    assert result['sensitivity']['rlaif']=={'S_reason':1.,'S_outcome':1.}
    assert len(result['rows'])==10 and len(pairs)==5
    assert all(r['tie_rate']==1 for r in result['rows'] if r['variant_type']=='clean_correct')
    assert signal_agreement(1,0,'B')==(False,'A')
