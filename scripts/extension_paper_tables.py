#!/usr/bin/env python
"""Verify saved extension predictions and generate the paper's extension tables."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/extensions'
LABELS={'gpt-j-6b':'GPT-J','llama-3.1-8b':'Llama-8B','gemma-2-9b':'Gemma base','gemma-2-9b-it':'Gemma IT'}
MAPPINGS={'natural':{'positive':'positive','negative':'negative'},'ab':{'positive':'A','negative':'B'},'ba':{'positive':'B','negative':'A'}}

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def score(p,t):
    assert len(p)==len(t) and len(t)>0
    return np.array(p)==np.array(t)
def verify_arm(arm):
    m=read(arm/'manifest.json');s=read(arm/'source_snapshot.json')
    assert hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()==m['source_sha256']
    return read(arm/'partitions.json'),m

def cell(path,split,method='tv',mapping=None):
    s=read(path);assert s['complete']
    sets={k:{p['input'] for p in v} for k,v in split.items()}
    assert not(sets['construction']&sets['dev'] or sets['construction']&sets['test'] or sets['dev']&sets['test'])
    originals={p['input'] for p in s['demos']}
    assert originals<=sets['construction'] and len(originals)==10
    assert s['dummy_query'] in sets['construction']-originals
    pool={(p['input'],mapping[p['output']] if mapping else p['output']) for p in split['construction']}
    assert all((p['input'],p['output']) in pool for p in s['demos'])
    assert read(path.parent/'selection.json')==s['selected']
    variants=[method,'fv_random_vector','fv_random_heads'] if method=='fv' else ['tv','tv_shuffle_0','tv_shuffle_1','tv_shuffle_2']
    assert set(s['selected'])==set(variants)
    for key,pred in s['predictions'].items():assert len(pred)==len(split[key.split('/')[0]])
    if mapping:
        for section in ['dev','test']:assert s['targets'][section]==[s['label_token_ids'][mapping[p['output']]] for p in split[section]]
    values={}
    for v in variants:
        dev={L:score(s['predictions'][f'dev/{v}/{L}'],s['targets']['dev']).mean() for L in s['layers']}
        assert s['selected'][v]==min(dev,key=lambda L:(-dev[L],L))
        values[v]=score(s['predictions'][f'test/{v}/{s["selected"][v]}'],s['targets']['test']).astype(float)
    for v in ['zero','icl']:values[v]=score(s['predictions'][f'test/{v}/None'],s['targets']['test']).astype(float)
    values['real']=values[method]
    values['control']=np.mean([values[v] for v in variants if v!=method],axis=0)
    values['difference']=values['real']-values['control']
    if method=='tv':
        ps=s.get('permutations',s.get('construction_details',{}).get('permutations'))
        for j,p in enumerate(ps):
            assert p['indices']==np.random.default_rng(s['seed']+10000+j).permutation(10).tolist()
            assert p['unchanged_labels']==sum(d['output']==s['demos'][i]['output'] for d,i in zip(s['demos'],p['indices']))
    preds=s['predictions'][f'test/{method}/{s["selected"][method]}']
    values['constant']=len(set(preds))==1
    if mapping:values['invalid']=np.mean([p not in s['label_token_ids'].values() for p in preds])
    return values,s

def aggregate(values,seed,strata=None):
    n=len(values['difference']);rng=np.random.default_rng(seed)
    indices=rng.integers(n,size=(10000,n)) if strata is None else np.concatenate([rng.choice(ix,size=(10000,len(ix)),replace=True) for ix in strata],axis=1)
    return {**{k:float(np.mean(v)*100) for k,v in values.items()},'ci95':(np.quantile(values['difference'][indices].mean(1),[.025,.975])*100).tolist()}

def analyze():
    result={'gemma_tv':{},'gemma_fv':{},'sentiment':{},'verified_cells':0}
    for kind in ['gemma_tv','gemma_fv']:
        prov=DATA/'provenance'/('gemma_tv' if kind=='gemma_tv' else 'gemma_fv')
        ex=read(prov/'execution.json')
        assert all(m['complete'] and m['pilot']['passed'] for m in ex['models'].values())
        for name,h in ex['source_hashes'].items():assert sha(prov/'source'/name)==h
        method='tv' if kind=='gemma_tv' else 'fv'
        for model in ['gemma-2-9b','gemma-2-9b-it']:
            arm=DATA/kind/model/method;parts,manifest=verify_arm(arm)
            expected=['antonym','country-capital'] if method=='fv' else ['antonym','country-capital','english-french','present-past','person-occupation','singular-plural','synonym','next_item']
            assert set(parts)==set(expected)
            old=read(ROOT/'results/heldout/llama-3.1-8b/tv/partitions.json')
            tasks={}
            for task in sorted(parts):
                assert parts[task]==old[task]
                assert sha(ROOT/f'data/tasks/{task}.json')==manifest['data_sha256'][task]
                cells=[]
                for seed in [100,101,102]:
                    v,s=cell(arm/task/str(seed)/'predictions.json',parts[task],method)
                    assert s['model']==model and s['seed']==seed and s['task']==task
                    cells.append(v);result['verified_cells']+=1
                tasks[task]={k:float(np.mean([c[k].mean() for c in cells])) for k in ['zero','icl','real','control','difference']}
            result[kind][model]={'tasks':tasks}
            if method=='tv':result[kind][model]['aggregate']=aggregate({k:np.array([t[k] for t in tasks.values()]) for k in ['zero','icl','real','control','difference']},20260905)
    base=result['gemma_tv']['gemma-2-9b']['tasks']
    tuned=result['gemma_tv']['gemma-2-9b-it']['tasks']
    result['gemma_tv_checkpoint_contrast']=aggregate(
        {'difference':np.array([tuned[t]['difference']-base[t]['difference'] for t in sorted(base)])},20260905)
    reference_split=None;matched={}
    for model in LABELS:
        group='sentiment_ndif' if model in ['gpt-j-6b','llama-3.1-8b'] else 'gemma_sentiment'
        arm=DATA/group/model;split,manifest=verify_arm(arm)
        assert sha(ROOT/'data/classification/sentiment.json')==manifest['data_sha256']
        if reference_split is None:reference_split=split
        assert split==reference_split
        assert len(split['test'])==80 and len(split['dev'])==40
        for section,n in [('test',40),('dev',20)]:
            assert all(sum(p['output']==label for p in split[section])==n for label in ['positive','negative'])
        conditions={};details={}
        for condition,mapping in MAPPINGS.items():
            cells=[];constants=0;invalid=[]
            for seed in [200,201,202]:
                v,s=cell(arm/condition/str(seed)/'predictions.json',split,mapping=mapping)
                assert s['model']==model and s['condition']==condition and s['seed']==seed
                pairing=([p['input'] for p in s['demos']],s['dummy_query'],s['permutations'])
                if seed in matched:assert matched[seed]==pairing
                else:matched[seed]=pairing
                cells.append(v);constants+=v['constant'];invalid.append(v['invalid']);result['verified_cells']+=1
            conditions[condition]={k:np.mean([c[k] for c in cells],axis=0) for k in ['zero','icl','real','control','difference']}
            details[condition]={'constant_cells':constants,'invalid_percent':float(np.mean(invalid)*100)}
        conditions['arbitrary']={k:(conditions['ab'][k]+conditions['ba'][k])/2 for k in conditions['ab']}
        strata=[np.array([i for i,p in enumerate(split['test']) if p['output']==label]) for label in ['negative','positive']]
        result['sentiment'][model]={c:aggregate(v,20260906,strata) for c,v in conditions.items()}
        result['sentiment'][model]['details']=details
    assert result['verified_cells']==96
    return result

def table(spec,header,rows):return '\n'.join([r'\begin{tabular}{'+spec+'}',r'\toprule',header+r' \\',r'\midrule',*[r+r' \\' for r in rows],r'\bottomrule',r'\end{tabular}',''])
def tables(s):
    gemma=[];sent=[];full=[];fvrows=[];taskrows=[]
    for model in ['gemma-2-9b','gemma-2-9b-it']:
        r=s['gemma_tv'][model]['aggregate'];lo,hi=r['ci95']
        gemma.append(f"{LABELS[model]} & {r['real']:.1f} & {r['control']:.1f} & ${r['difference']:+.1f}$ & $[{lo:.1f},{hi:.1f}]$")
        for task,r in s['gemma_fv'][model]['tasks'].items():fvrows.append(f"{LABELS[model]} & {task} & {100*r['icl']:.1f} & {100*r['real']:.1f} & {100*r['control']:.1f}")
    for model in LABELS:
        r=s['sentiment'][model]['arbitrary'];sent.append(f"{LABELS[model]} & {r['icl']:.1f} & {r['real']:.1f} & {r['control']:.1f}")
        for c,label in [('natural','Natural'),('ab','A/B'),('ba','B/A')]:
            r=s['sentiment'][model][c];lo,hi=r['ci95'];detail=s['sentiment'][model]['details'][c]
            full.append(f"{LABELS[model]} & {label} & {r['zero']:.1f} & {r['icl']:.1f} & {r['real']:.1f} & {r['control']:.1f} & ${r['difference']:+.1f}$ & $[{lo:.1f},{hi:.1f}]$ & {detail['constant_cells']}/3")
    for task in sorted(s['gemma_tv']['gemma-2-9b']['tasks']):
        a=s['gemma_tv']['gemma-2-9b']['tasks'][task];b=s['gemma_tv']['gemma-2-9b-it']['tasks'][task]
        taskrows.append(task.replace('_','-')+' & '+' & '.join(f'{100*r[k]:.1f}' for r in [a,b] for k in ['icl','real','control']) )
    return {'gemma_heldout_table.tex':table('lrrrr',r'Model & TV & Control & $\Delta$ & 95\% CI',gemma),
            'sentiment_summary_table.tex':table('lrrr','Model & ICL & TV & Control',sent),
            'sentiment_full_table.tex':table('llrrrrrrr',r'Model & Labels & Zero & ICL & TV & Control & $\Delta$ & 95\% CI & Constant',full),
            'gemma_fv_table.tex':table('llrrr','Model & Task & ICL & FV & Control',fvrows),
            'gemma_task_table.tex':table('lrrrrrr','Task & Base ICL & Base TV & Base ctrl. & IT ICL & IT TV & IT ctrl.',taskrows)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    summary=analyze()
    for name,content in tables(summary).items():
        path=ROOT/'paper/generated'/name
        if args.check:assert path.read_text()==content,f'Stale table: {name}'
        else:path.write_text(content)
    path=DATA/'summary.json'
    if args.check:assert read(path)==summary,'Stale extension summary'
    else:path.write_text(json.dumps(summary,indent=2)+'\n')
    print('Verified 96 extension cells and five generated tables.')
