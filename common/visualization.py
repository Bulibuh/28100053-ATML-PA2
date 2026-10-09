"""Plot saved course results; never fill missing experiments with simulated metrics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common.data import repo_path,read_jsonl


def read(path):return json.loads(Path(path).read_text())


def flat_metrics(value,prefix=''):
    out={}
    for key,item in value.items():
        if key in ('config','ids'):continue
        name=f'{prefix}.{key}' if prefix else key
        if isinstance(item,dict):out.update(flat_metrics(item,name))
        elif isinstance(item,(int,float)) or item is None:out[name]=item
    return out


def metric_table(directory):
    rows=[]
    for path in sorted(Path(directory).glob('*/metrics.json')):
        value=read(path)
        rows.append({'condition':path.parent.name,**flat_metrics(value)})
    return pd.DataFrame(rows)


def save_figure(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout();fig.savefig(path,dpi=160,bbox_inches='tight')
    return path


def show_table(frame):
    from IPython.display import display
    display(frame)


def bars(frame,columns,title,path,index='condition'):
    columns=[c for c in columns if c in frame and frame[c].notna().any()]
    if frame.empty or not columns:return
    fig,axes=plt.subplots(1,len(columns),figsize=(5*len(columns),4),squeeze=False)
    for ax,column in zip(axes.flat,columns):
        ax.bar(frame[index].astype(str),frame[column]);ax.set_title(column);ax.tick_params(axis='x',rotation=45)
    fig.suptitle(title);save_figure(fig,path);plt.show()


def trajectories(directory,columns,title,path):
    data={p.parent.name:pd.DataFrame(read_jsonl(p)) for p in sorted(Path(directory).glob('*/training.jsonl'))}
    if not data:return
    columns=[c for c in columns if any(c in d for d in data.values())]
    fig,axes=plt.subplots(int(np.ceil(len(columns)/3)),3,figsize=(15,4*int(np.ceil(len(columns)/3))),squeeze=False)
    for ax,column in zip(axes.flat,columns):
        for name,df in data.items():
            if column in df:ax.plot(df['update'] if 'update' in df else df['step'],df[column],label=name)
        ax.set_title(column);ax.set_xlabel('update / optimizer step');ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(columns):]:ax.set_visible(False)
    fig.suptitle(title);save_figure(fig,path);plt.show()


def examples(directory,limit=3):
    paths=sorted(Path(directory).glob('*/generations.jsonl'))+sorted(Path(directory).glob('examples.jsonl'))+sorted(Path(directory).glob('gsm/comparisons.jsonl'))
    for path in paths:
        rows=read_jsonl(path)
        if not rows:continue
        print('\nSaved examples:',path.relative_to(repo_path('.')))
        for row in rows[:limit]:
            print({k:row[k] for k in ('source_index','prompt_id','xstest_id','policy','reward','judge_preference','verifier_preference','label') if k in row})
            print('Prompt:',str(row.get('prompt',row.get('question','')))[:200])
            print('Response:',str(row.get('response',row.get('candidate','')))[:700])
        break


def plot_task(task,results='results'):
    names={1:'task1_dpo',2:'task2_ppo',3:'task3_grpo',4:'task4_safety',5:'task5_feedback'}
    directory=repo_path(results)/names[task];figdir=directory/'figures'
    if not directory.exists():
        print('Course results have not been generated for this task. No course metrics are inferred.');return
    if task<=3:
        table=metric_table(directory);show_table(table)
        bars(table,['preference_accuracy','reward_mean','kl','response_length_mean'],f'Task {task}: held-out course results',figdir/'held_out.png')
        columns={1:['loss','preference_accuracy','gradient_norm'],2:['reward','kl','policy_loss','value_loss','entropy','clip_fraction','gradient_norm','response_length'],
                 3:['reward','kl','reward_std','uninformative_fraction','policy_loss','gradient_norm','entropy','response_length']}[task]
        trajectories(directory,columns,f'Task {task}: optimization diagnostics',figdir/'trajectories.png')
        resources=[]
        for path in sorted(directory.glob('*/run.json')):
            metadata=read(path);resources.append({'condition':path.parent.name,**{k:metadata.get(k) for k in ('completed','wall_seconds','peak_vram_bytes','generated_tokens','gradient_norm_std','examples_seen','optimizer_steps')}})
        if resources:show_table(pd.DataFrame(resources))
        if task==1 and not table.empty:
            bars(table,[c for c in table if c.startswith('strata.') and c.endswith('preference_accuracy')],'Length-stratified preference accuracy',figdir/'length_strata.png')
            bars(table,['word_limits.word_limit_compliance','word_limits.response_length_mean'],'Common word-limit prompts',figdir/'word_limits.png')
        if task==2 and (directory/'cached_clipping.json').exists():
            df=pd.DataFrame(read(directory/'cached_clipping.json')['rows']);show_table(df)
            bars(df,['clipped_surrogate','affected_token_fraction'],'Cached PPO clipping',figdir/'clipping.png','epsilon')
        if task==3 and (directory/'group_size.json').exists():
            df=pd.DataFrame(read(directory/'group_size.json')['rows']);show_table(df)
            fig,axes=plt.subplots(1,3,figsize=(14,4))
            for ax,column in zip(axes,['informative_fraction','reward_std','advantage_variance']):
                for difficulty,chunk in df.groupby('difficulty'):
                    if column in chunk:ax.plot(chunk['k'],chunk[column],marker='o',label=difficulty)
                ax.set_title(column);ax.set_xlabel('K');ax.set_xticks([2,4,8]);ax.legend()
            save_figure(fig,figdir/'group_sizes.png');plt.show()
        if task==3:
            files=sorted(directory.glob('normalization_*/length_gradients.jsonl'))
            if files:
                fig,ax=plt.subplots(figsize=(8,4));frames=[]
                for path in files:
                    df=pd.DataFrame(read_jsonl(path));df=df[df['tokens']>0];df['condition']=path.parent.name;frames.append(df)
                    ax.scatter(df['tokens'],df['logprob_gradient_per_token'],s=12,alpha=.6,label=path.parent.name)
                ax.set(xlabel='Valid sequence tokens',ylabel='Absolute dL/d log π per token');ax.legend()
                save_figure(fig,figdir/'length_gradient.png');plt.show()
                all_rows=pd.concat(frames,ignore_index=True)
                threshold=all_rows['tokens'].median();all_rows['length_bin']=np.where(all_rows['tokens']<=threshold,'short','long')
                print('Shared short/long threshold:',threshold)
                show_table(all_rows.groupby(['condition','length_bin'])[['logprob_gradient_l1','logprob_gradient_per_token','tokens']].mean().reset_index())
    if task==4 and (directory/'metrics.json').exists():
        metrics=read(directory/'metrics.json');df=pd.DataFrame([{'condition':p,**flat_metrics(m)} for p,m in metrics.items()]);show_table(df)
        bars(df,['safe_answer_rate','safe_over_refusal_rate','unsafe_compliance_rate','justified_refusal_rate','ambiguous_rate'],'Safety calibration',figdir/'safety.png')
        categories=read(directory/'categories.json');catdf=pd.DataFrame(categories);show_table(catdf.drop(columns=['label_counts']))
        counts=pd.DataFrame([{'policy':r['policy'],'category':r['category'],**r['label_counts']} for r in categories]).fillna(0)
        cols=[c for c in counts if c not in ('policy','category')]
        fig,axes=plt.subplots(2,2,figsize=(16,10))
        for ax,policy in zip(axes.flat,metrics):
            chunk=counts[counts['policy']==policy].set_index('category')[cols];chunk.div(chunk.sum(axis=1),axis=0).plot.bar(stacked=True,ax=ax)
            ax.set_title(policy);ax.set_ylabel('Judge-label fraction')
        save_figure(fig,figdir/'categories.png');plt.show()
        if (directory/'audit_metrics.json').exists():
            audit=read(directory/'audit_metrics.json');print('Manual audit complete:',audit['complete'],'Labeled:',audit['labeled'],'of',audit['expected'])
            if audit['labeled']:
                labels=['SAFE_ANSWER','JUSTIFIED_REFUSAL','UNSAFE_COMPLIANCE','OVER_REFUSAL','AMBIGUOUS']
                fig,axes=plt.subplots(2,2,figsize=(16,12))
                for ax,(policy,stats) in zip(axes.flat,audit['policies'].items()):
                    matrix=np.array([[stats['confusion'][a].get(b,0) for b in labels] for a in labels]);ax.imshow(matrix)
                    ax.set_xticks(range(5),labels,rotation=60);ax.set_yticks(range(5),labels);ax.set_title(f"{policy}: agreement {stats['agreement']}")
                    for i in range(5):
                        for j in range(5):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='red')
                save_figure(fig,figdir/'manual_confusion.png');plt.show()
    if task==5:
        tables=[]
        for dataset in ('gsm','transfer'):
            path=directory/dataset/'metrics.json'
            if path.exists():
                df=pd.DataFrame([{'condition':p,'dataset':dataset,**m} for p,m in read(path).items()]);tables.append(df);show_table(df)
                bars(df,['exact_accuracy','pairwise_win_rate','format_compliance','response_length_mean'],dataset,figdir/f'{dataset}.png')
        path=directory/'diagnostics'/'metrics.json'
        if path.exists():
            metrics=read(path);df=pd.DataFrame(metrics['rows']);show_table(df);show_table(pd.DataFrame(metrics['sensitivity']).T)
            fig,axes=plt.subplots(1,2,figsize=(16,5))
            for ax,mechanism in zip(axes,['rlvr','rlaif']):
                chunk=df[df['mechanism']==mechanism].set_index('variant_type')[['better_rate','tie_rate','wrong_rate']]
                chunk.plot.bar(stacked=True,ax=ax);ax.set_title(mechanism)
            save_figure(fig,figdir/'diagnostics.png');plt.show()
            for pair in read_jsonl(directory/'diagnostics'/'pairs.jsonl')[:5]:
                print(pair['variant_type'],pair['verifier_preference'],pair['ai_preference']);print(pair['variant'][:600])
        if (directory/'comparison.json').exists():show_table(pd.DataFrame(read(directory/'comparison.json')['rows']))
    examples(directory)
