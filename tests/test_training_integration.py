"""Run the real loops on tiny random Qwen models/LoRA, with no network or course metrics."""
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import Qwen2Config,Qwen2ForCausalLM,Qwen2ForSequenceClassification
from peft import LoraConfig,get_peft_model
from common.data import load_yaml
from common.models import value_parameter_groups
from common.experiment_utils import Run


class TinyTokenizer:
    pad_token_id=0
    eos_token_id=2
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False,**kwargs):
        text=' '.join(str(m['content']) for m in messages)
        return [1,3,4] if tokenize else text
    def __call__(self,texts,return_tensors=None,padding=False,truncation=False,max_length=None,add_special_tokens=False,**kwargs):
        if isinstance(texts,str):return {'input_ids':[3+ord(c)%26 for c in texts]}
        ids=[[1,3]+[3+ord(c)%26 for c in text][:3] for text in texts]
        width=max(map(len,ids));mask=[[0]*(width-len(row))+[1]*len(row) for row in ids];ids=[[0]*(width-len(row))+row for row in ids]
        return {'input_ids':torch.tensor(ids),'attention_mask':torch.tensor(mask)}
    def decode(self,row,skip_special_tokens=True):return ' '.join(str(int(x)) for x in row if int(x) not in (0,1,2))
    def save_pretrained(self,path):Path(path).mkdir(parents=True,exist_ok=True);(Path(path)/'tiny_tokenizer.json').write_text('{}')


def qwen_config():
    return Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,
                       max_position_embeddings=64,pad_token_id=0,eos_token_id=2,bos_token_id=1,attention_dropout=0.)


def policy():
    model=get_peft_model(Qwen2ForCausalLM(qwen_config()),LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    return model


def reward():
    cfg=qwen_config();cfg.num_labels=1
    model=Qwen2ForSequenceClassification(cfg).eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


def critic():
    cfg=qwen_config();cfg.num_labels=1
    model=Qwen2ForSequenceClassification(cfg)
    return get_peft_model(model,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='SEQ_CLS',modules_to_save=['score']))


@pytest.fixture
def local_run(tmp_path,monkeypatch):
    torch.set_num_threads(1);torch.manual_seed(22)
    import shutil
    shutil.copytree(Path(__file__).resolve().parents[1]/'configs',tmp_path/'configs')
    import common.data as data
    monkeypatch.setattr(data,'REPO_ROOT',tmp_path)
    return tmp_path


def test_dpo_actual_lora_update_and_partial_accumulation(local_run,monkeypatch):
    import task1_dpo.experiments as exp
    from task1_dpo.train import make_collate
    cfg=load_yaml(Path(__file__).resolve().parents[1]/'configs/dpo.yaml')
    cfg.update(batch_size=1,grad_accum_steps=2,epochs=1,max_sequence_length=12)
    rows=[{'source_index':i,'prompt':'x','chosen':'ab','rejected':'cd'} for i in range(3)]
    model=policy();tok=TinyTokenizer();params=[p for p in model.parameters() if p.requires_grad];before=[p.detach().clone() for p in params]
    bundle={'cfg':cfg,'rows':rows,'model':model,'tokenizer':tok,'optimizer':AdamW(params,lr=.01),
            'loader':DataLoader(rows,batch_size=1,collate_fn=make_collate(tok,12)),'beta':.1}
    monkeypatch.setattr(exp,'prepare_dpo_run',lambda *args:bundle)
    out=exp.train(name='unit_dpo')
    assert (Path(out)/'adapter_model.safetensors').exists()
    assert any(not torch.equal(a,b) for a,b in zip(before,params))
    metadata=json.loads((local_run/'results/task1_dpo/unit_dpo/run.json').read_text())
    assert metadata['optimizer_steps']==2 and metadata['examples_seen']==3 and metadata['completed']
    with pytest.raises(FileExistsError):exp.train(name='unit_dpo')


def test_ppo_actual_policy_and_critic_updates(local_run,monkeypatch):
    import task2_ppo.experiments as exp
    cfg=load_yaml(Path(__file__).resolve().parents[1]/'configs/ppo.yaml');cfg.update(updates=2,max_response_length=3,ppo_epochs=2)
    p=policy();v=critic();tok=TinyTokenizer();params=[x for x in p.parameters() if x.requires_grad];vp=[x for x in v.parameters() if x.requires_grad]
    before=[x.detach().clone() for x in params];vb=[x.detach().clone() for x in vp]
    bundle={'cfg':cfg,'policy':p,'value_model':v,'tokenizer':tok,'reward_model':reward(),'reward_tokenizer':tok,
            'prompt_rows':[{'source_index':1,'prompt':'x'}],'policy_optimizer':AdamW(params,lr=.01),
            'value_optimizer':AdamW(value_parameter_groups(v,.01,.02))}
    monkeypatch.setattr(exp,'prepare_ppo_continuation',lambda *args:bundle)
    out=exp.train(name='unit_ppo')
    assert (Path(out)/'value/adapter_model.safetensors').exists()
    assert any(not torch.equal(a,b) for a,b in zip(before,params))
    assert any(not torch.equal(a,b) for a,b in zip(vb,vp))
    records=[json.loads(line) for line in (local_run/'results/task2_ppo/unit_ppo/training.jsonl').read_text().splitlines()]
    assert len(records)==2 and all(r['generated_tokens']>0 for r in records)
    assert all(r['value_loss'] is not None and r['gradient_norm'] is not None for r in records)


@pytest.mark.parametrize('loss_type',['grpo','dr_grpo'])
def test_grpo_actual_group_rollout_and_checkpoint(local_run,monkeypatch,loss_type):
    import task3_grpo.experiments as exp
    cfg=load_yaml(Path(__file__).resolve().parents[1]/'configs/grpo.yaml');cfg.update(updates=2,max_completion_length=4,mask_truncated_completions=False)
    p=policy();tok=TinyTokenizer();params=[x for x in p.parameters() if x.requires_grad]
    before=[x.detach().clone() for x in params]
    bundle={'cfg':cfg,'policy':p,'tokenizer':tok,'reward_model':reward(),'reward_tokenizer':tok,
            'prompt_rows':[{'source_index':1,'prompt':'x'}],'optimizer':AdamW(params,lr=.01)}
    monkeypatch.setattr(exp,'prepare_grpo_continuation',lambda *args:bundle)
    out=exp.train(name='unit_'+loss_type,loss_type=loss_type)
    assert (Path(out)/'adapter_model.safetensors').exists()
    assert any(not torch.equal(a,b) for a,b in zip(before,params))
    records=[json.loads(line) for line in (local_run/f'results/task3_grpo/unit_{loss_type}/length_gradients.jsonl').read_text().splitlines()]
    assert len(records)==8
    assert all(r['logprob_gradient_per_token'] is not None for r in records)
