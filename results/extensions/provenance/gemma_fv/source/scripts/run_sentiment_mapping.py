#!/usr/bin/env python
"""Paired sentiment TV experiment. Pilot never evaluates test inputs."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fvtv import tasks, tv, fv, eval_icl, heldout
from run_grid import load_model, MODEL_REGISTRY
from run_heldout import write_json

MAPPINGS = {'natural': {'positive':'positive','negative':'negative'},
            'ab': {'positive':'A','negative':'B'},
            'ba': {'positive':'B','negative':'A'}}
SEEDS = [200,201,202]


def partition(rows):
    groups = {}
    for row in rows:
        if row['output'] not in MAPPINGS['natural']:
            raise ValueError('Unexpected sentiment label')
        groups.setdefault(row['input'], set()).add(row['output'])
    conflicts = sorted(x for x, ys in groups.items() if len(ys)!=1)
    clean = [{'input':x,'output':next(iter(ys))} for x,ys in sorted(groups.items()) if len(ys)==1]
    split = {k:[] for k in ['construction','dev','test']}
    rng = np.random.default_rng(20260906)
    for label in ['negative','positive']:
        pool = [p for p in clean if p['output']==label]
        order = rng.permutation(len(pool))
        for name, indices in [('test',order[:40]),('dev',order[40:60]),('construction',order[60:])]:
            split[name].extend(pool[i] for i in indices)
    for name in split:
        rng.shuffle(split[name])
    return split, {'raw_rows':len(rows),'unique_inputs':len(groups),'conflicting_inputs_excluded':conflicts,'retained_inputs':len(clean)}


def demonstrations(pool, seed):
    rng = np.random.default_rng(seed)
    demos = []
    for label in ['negative','positive']:
        candidates = [p for p in pool if p['output']==label]
        demos.extend(candidates[i] for i in rng.choice(len(candidates),5,replace=False))
    rng.shuffle(demos)
    return demos


def mapped(rows, condition):
    return [dict(input=p['input'],output=MAPPINGS[condition][p['output']]) for p in rows]


def check_tokens(tokenizer, prompts, labels):
    ids = {}
    for label in labels:
        target = tokenizer.encode(' '+label, add_special_tokens=False)
        if len(target)!=1:
            raise ValueError(f'Label is not single-token: {label}')
        for prompt in prompts:
            prefix = tokenizer.encode(prompt, add_special_tokens=False)
            full = tokenizer.encode(prompt+' '+label, add_special_tokens=False)
            if full != prefix+target:
                raise ValueError(f'Context changes target token: {label}')
        ids[label] = target[0]
    if len(set(ids.values()))!=len(ids):
        raise ValueError('Label tokens collide')
    return ids


def run(args):
    out = Path(args.out)
    paths = sorted((ROOT/'src/fvtv').glob('*.py'))+[Path(__file__),ROOT/'scripts/run_grid.py',ROOT/'scripts/run_heldout.py']
    source = {str(p.relative_to(ROOT)):p.read_text() for p in paths}
    manifest = {'model':args.model,'seeds':SEEDS,'mappings':MAPPINGS,'split_seed':20260906,
                'dev_per_class':20,'test_per_class':40,'demos_per_class':5,'batch_size':4,
                'source_sha256':heldout.digest(source),'data_sha256':hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                'analysis':'Unrestricted top-1 accuracy; invalid-label rate; independently dev-selected controls; lowest-layer tie break. Average seeds and then arbitrary mappings within each test input. Paired stratified input bootstrap, 10000 draws, RNG 20260906. Report both mappings separately too.'}
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())!=manifest:
            raise ValueError('Resume refused: protocol/source/data changed')
    else:
        write_json(out/'manifest.json',manifest)
        write_json(out/'source_snapshot.json',source)
    split, audit = partition(json.loads(Path(args.data).read_text()))
    write_json(out/'partitions.json',split)
    write_json(out/'data_audit.json',audit)
    model = load_model(MODEL_REGISTRY[args.model]['hf_id'],True)
    fv.verify_arch(model,remote=True)
    cfg = fv.arch_config(model)
    layers = sorted(set(range(0,cfg['n_layers'],4))|{cfg['n_layers']-1})
    for seed in (SEEDS[:1] if args.phase=='pilot' else SEEDS):
        raw_demos = demonstrations(split['construction'],seed)
        dummy = tv.pick_dummy_query(split['construction'],raw_demos,seed+555)
        for condition in MAPPINGS:
            cell = out/condition/str(seed)
            sf = cell/'predictions.json'
            state = json.loads(sf.read_text()) if sf.exists() else {'predictions':{},'selected':{}}
            if state.get('complete') or (args.phase=='pilot' and state.get('pilot_complete')):
                continue
            demos = mapped(raw_demos,condition)
            subsets = {s:mapped(rows,condition) for s,rows in split.items()}
            # Prompt/token checks inspect strings only; no test inference or scoring.
            all_prompts = [tasks.build_icl_prompt(demos,p['input']) for p in split['dev']+split['test']]
            all_prompts += [tasks.build_icl_prompt(demos,dummy)]
            all_prompts += [tasks.build_zeroshot_prompt(p['input']) for p in split['dev']+split['test']]
            token_ids = check_tokens(model.tokenizer,all_prompts,list(MAPPINGS[condition].values()))
            lengths = [len(model.tokenizer.encode(p)) for p in all_prompts]
            limit = getattr(model.config,'max_position_embeddings',getattr(model.config,'n_positions',2048))
            if max(lengths)>limit:
                raise ValueError(f'Prompt exceeds context: {max(lengths)} > {limit}')
            state.update(model=args.model,seed=seed,condition=condition,demos=demos,dummy_query=dummy,layers=layers,
                         label_token_ids=token_ids,max_prompt_tokens=max(lengths),
                         targets={s:[token_ids[p['output']] for p in subsets[s]] for s in ['dev','test']})
            write_json(sf,state)
            print(f'{time.ctime()} {condition}/{seed}: construction; max prompt {max(lengths)}',flush=True)
            af = cell/'vectors.pt'
            if af.exists():
                artifacts = torch.load(af,map_location='cpu',weights_only=True)
            else:
                artifacts = {'vectors':{},'permutations':[]}
                artifacts['vectors']['tv'] = tv.extract_theta_all_layers(model,demos,dummy,remote=True)
                for j in range(3):
                    order = np.random.default_rng(seed+10000+j).permutation(10).tolist()
                    shuffled = [dict(input=p['input'],output=demos[k]['output']) for p,k in zip(demos,order)]
                    artifacts['permutations'].append({'indices':order,'unchanged_labels':sum(p['output']==q['output'] for p,q in zip(demos,shuffled))})
                    artifacts['vectors'][f'tv_shuffle_{j}'] = tv.extract_theta_all_layers(model,shuffled,dummy,remote=True)
                torch.save(artifacts,af.with_suffix('.tmp'))
                af.with_suffix('.tmp').replace(af)
            state['permutations'] = artifacts['permutations']
            vectors = artifacts['vectors']
            def predict(section, variant, layer=None):
                if args.phase=='pilot' and section=='test':
                    raise AssertionError('Pilot test leakage')
                key = f'{section}/{variant}/{layer}'
                if key not in state['predictions']:
                    pairs = subsets[section]
                    prompts = [tasks.build_icl_prompt(demos,p['input']) if variant=='icl' else tasks.build_zeroshot_prompt(p['input']) for p in pairs]
                    if variant in ['zero','icl']:
                        preds = eval_icl.predict_top1(model,prompts,batch_size=4,remote=True)
                    else:
                        preds = tv.patch_theta(model,prompts,vectors[variant][layer],layer,batch_size=4,remote=True)
                    if len(preds)!=len(pairs):
                        raise ValueError('Incomplete predictions')
                    state['predictions'][key] = preds
                    write_json(sf,state)
                    print(f'{time.ctime()} {condition}/{seed}: saved {key}',flush=True)
                return state['predictions'][key]
            for baseline in ['zero','icl']:
                predict('dev',baseline)
            for variant in vectors:
                chosen = heldout.select_layer({L:predict('dev',variant,L) for L in layers},state['targets']['dev'])
                if variant in state['selected'] and chosen!=state['selected'][variant]:
                    raise ValueError('Frozen layer changed')
                state['selected'][variant]=chosen
            write_json(cell/'selection.json',state['selected'])
            state['pilot_complete']=True
            write_json(sf,state)
            if args.phase=='pilot':
                print(f'{time.ctime()} {condition}/{seed}: PILOT COMPLETE',flush=True)
                continue
            for baseline in ['zero','icl']:
                predict('test',baseline)
            for variant in vectors:
                predict('test',variant,state['selected'][variant])
                if variant!='tv':
                    predict('test',variant,state['selected']['tv'])
            state['complete']=True
            state['completed_at_unix']=time.time()
            write_json(sf,state)
            print(f'{time.ctime()} {condition}/{seed}: COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['gpt-j-6b','llama-3.1-8b','gemma-2-9b','gemma-2-9b-it'],required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--phase',choices=['pilot','full'],required=True)
    run(p.parse_args())
