#!/usr/bin/env python
"""Prospective construction/development/test evaluation; original grid is untouched."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fvtv import controls, eval_icl, fv, heldout, tasks, tv
from run_grid import load_model, MODEL_REGISTRY
import numpy as np
import torch


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def run(args):
    out = Path(args.out)
    task_names = args.tasks.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    source_paths = sorted((ROOT / 'src/fvtv').glob('*.py')) + [Path(__file__), ROOT/'scripts/run_grid.py']
    source = {str(p.relative_to(ROOT)): p.read_text() for p in source_paths}
    config = vars(args).copy()
    config.pop('out')
    manifest = {'config': config, 'source_sha256': heldout.digest(source),
                'data_sha256': {t: hashlib.sha256((tasks.TASKS_DIR/f'{t}.json').read_bytes()).hexdigest() for t in task_names}}
    path = out / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Resume refused: source, arguments or data changed; use a new output directory')
    else:
        write_json(path, manifest)
        write_json(out/'source_snapshot.json', source)
    partitions = {t: heldout.partition(tasks.load_task(t)) for t in task_names}
    write_json(out/'partitions.json', partitions)
    print(f'Loading {args.model}; {args.method}; {len(task_names)*len(seeds)} cells', flush=True)
    model = load_model(MODEL_REGISTRY[args.model]['hf_id'], True)
    fv.verify_arch(model, remote=True)
    cfg = fv.arch_config(model)
    layers = sorted(set(range(0, cfg['n_layers'], 4)) | {cfg['n_layers']-1})
    for name in task_names:
        for seed in seeds:
            cell_dir = out / name / str(seed)
            state_file = cell_dir/'predictions.json'
            state = json.loads(state_file.read_text()) if state_file.exists() else {'predictions': {}, 'selected': {}}
            if state.get('complete'):
                print(f'{name} {seed}: complete; skipped', flush=True)
                continue
            split = partitions[name]
            construction = split['construction']
            demos = tasks.sample_demos(construction, seed)
            dummy = tv.pick_dummy_query(construction, demos, seed + 555)
            metadata = {'task': name, 'seed': seed, 'model': args.model, 'method': args.method,
                        'demos': demos, 'dummy_query': dummy, 'layers': layers,
                        'targets': {s: [tasks.target_first_token_id(model.tokenizer, p['output']) for p in split[s]] for s in ['dev','test']}}
            state.update(metadata)
            write_json(state_file, state)
            artifact_file = cell_dir/'vectors.pt'
            print(f'{name} {seed}: constructing {args.method}', flush=True)
            if artifact_file.exists():
                artifacts = torch.load(artifact_file, map_location='cpu', weights_only=True)
            elif args.method == 'tv':
                vectors = {'tv': tv.extract_theta_all_layers(model, demos, dummy, remote=True)}
                permutations = []
                for j in range(3):
                    permutation = np.random.default_rng(seed+10000+j).permutation(len(demos)).tolist()
                    shuffled = [dict(input=d['input'], output=demos[k]['output']) for d,k in zip(demos,permutation)]
                    permutations.append({'indices': permutation, 'unchanged_labels': sum(d['output']==s['output'] for d,s in zip(demos,shuffled))})
                    vectors[f'tv_shuffle_{j}'] = tv.extract_theta_all_layers(model, shuffled, dummy, remote=True)
                artifacts = {'vectors': vectors, 'permutations': permutations}
                torch.save(artifacts, artifact_file)
            else:
                mean_file = cell_dir/'mean.pt'
                if mean_file.exists():
                    means = torch.load(mean_file, map_location='cpu', weights_only=True)
                else:
                    means = fv.compute_mean_head_activations(model, construction, seed, n_ex=32, remote=True)
                    torch.save(means, mean_file)
                aie_file = cell_dir/'aie.pt'
                if aie_file.exists():
                    aie = torch.load(aie_file, map_location='cpu', weights_only=True)
                else:
                    aie = fv.compute_aie(model, construction, means, seed, n_trials=10, remote=True, max_batch_rows=32)
                    torch.save(aie, aie_file)
                top = fv.top_k_heads(aie, k=10)
                random_heads = controls.pick_random_heads(cfg, 10, seed=seed)
                params = fv.grab_out_proj_params(model, cfg, layers={h[0] for h in top+random_heads}, remote=True)
                vector = fv.compute_fv(model, means, top, out_proj_params=params)
                vectors = {'fv': vector, 'fv_random_vector': controls.random_vector(vector, seed=seed),
                           'fv_random_heads': fv.compute_fv(model, means, random_heads, out_proj_params=params)}
                artifacts = {'vectors': vectors, 'top_heads': top, 'random_heads': random_heads}
                torch.save(artifacts, artifact_file)
            state['construction_details'] = {k:v for k,v in artifacts.items() if k!='vectors'}
            vectors = artifacts['vectors']
            def predict(split_name, variant, layer=None):
                key = f'{split_name}/{variant}/{layer}'
                if key not in state['predictions']:
                    pairs = split[split_name]
                    prompts = [tasks.build_zeroshot_prompt(p['input']) for p in pairs]
                    if variant == 'icl':
                        prompts = [tasks.build_icl_prompt(demos,p['input']) for p in pairs]
                    if variant in ['icl','zero']:
                        preds = eval_icl.predict_top1(model,prompts,batch_size=16,remote=True)
                    elif args.method == 'tv':
                        preds = tv.patch_theta(model,prompts,vectors[variant][layer],layer,batch_size=16,remote=True)
                    else:
                        preds = fv.inject_fv(model,prompts,vectors[variant],layer,batch_size=16,remote=True)
                    if len(preds)!=len(pairs):
                        raise ValueError('Incomplete predictions')
                    state['predictions'][key] = preds
                    write_json(state_file,state)
                    print(f'{name} {seed}: saved {key}',flush=True)
                return state['predictions'][key]
            # Complete every development sweep and freeze choices before any test call.
            for variant in vectors:
                dev_predictions = {L: predict('dev',variant,L) for L in layers}
                chosen = heldout.select_layer(dev_predictions,state['targets']['dev'])
                if variant in state['selected'] and state['selected'][variant]!=chosen:
                    raise ValueError('Previously frozen layer changed')
                state['selected'][variant]=chosen
            write_json(state_file,state)
            write_json(cell_dir/'selection.json',state['selected'])
            for baseline in ['zero','icl']:
                predict('test',baseline)
            for variant in vectors:
                predict('test',variant,state['selected'][variant])
                if variant != args.method:
                    predict('test',variant,state['selected'][args.method])
            state['complete']=True
            state['completed_at_unix']=time.time()
            write_json(state_file,state)
            print(f'{name} {seed}: COMPLETE',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['gpt-j-6b','llama-3.1-8b','gemma-2-9b','gemma-2-9b-it'],required=True)
    p.add_argument('--method',choices=['tv','fv'],required=True)
    p.add_argument('--tasks',default=','.join(heldout.TASK_NAMES))
    p.add_argument('--seeds',default='100,101,102')
    p.add_argument('--out',required=True)
    run(p.parse_args())
