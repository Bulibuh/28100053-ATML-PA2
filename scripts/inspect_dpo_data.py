"""Audit fixed DPO prompt lengths before an expensive training run."""
import argparse
from transformers import AutoTokenizer
from common.data import load_yaml,read_jsonl,repo_path,prompt_messages_from_preference,preference_responses,encode_prompt_response
from common.experiment_utils import dump,identity


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--max-sequence-length',type=int,default=768);args=ap.parse_args()
    cfg=load_yaml('configs/base.yaml')
    # The pinned adapter's non-LFS vocab/merges permit local fast-tokenizer construction.
    tokenizer=AutoTokenizer.from_pretrained(str(repo_path(cfg['paths']['grpo_midpoint_policy'])),local_files_only=True,use_fast=True)
    reports=[]
    for key in ('dpo_standard_train','dpo_length_train','dpo_standard_eval','dpo_length_eval'):
        rows=read_jsonl(cfg['paths'][key]);bad=[];maximum=0
        for row in rows:
            messages=prompt_messages_from_preference(row)
            n=len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True));maximum=max(maximum,n)
            if n>=args.max_sequence_length:bad.append({**identity(row),'prompt_tokens':n});continue
            for response in preference_responses(row):
                ids,mask=encode_prompt_response(tokenizer,messages,response,args.max_sequence_length)
                if tokenizer(response,add_special_tokens=False)['input_ids'] and sum(mask)<=1:
                    raise ValueError('Only EOS fits for a nonempty response; increase the runtime sequence limit')
        reports.append({'dataset':cfg['paths'][key],'examples':len(rows),'maximum_prompt_tokens':maximum,'rejected_prompt_count':len(bad),'rejected_ids':bad})
        print(key,len(rows),'examples; max prompt',maximum,'; rejected',len(bad),'at cap',args.max_sequence_length)
    dump(f'results/validation/dpo_data_{args.max_sequence_length}.json',{'max_sequence_length':args.max_sequence_length,'rows':reports})
    if any(r['rejected_prompt_count'] for r in reports):raise SystemExit('Fixed prompts exceed the cap. Choose an explicit prompt-preserving override; no rows were altered or filtered.')


if __name__=='__main__':main()
