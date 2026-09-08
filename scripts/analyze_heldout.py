#!/usr/bin/env python
"""Independently verify saved held-out predictions and summarize completed cells."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def accuracy(pred, target):
    if len(pred)!=len(target) or not target:
        raise ValueError('Invalid prediction length')
    return sum(int(a==b) for a,b in zip(pred,target))/len(target)


def analyze(root):
    rows=[]
    coverage=[]
    for manifest_path in sorted(root.glob('*/*/manifest.json')):
        arm=manifest_path.parent
        manifest=json.loads(manifest_path.read_text())
        snapshot=json.loads((arm/'source_snapshot.json').read_text())
        assert hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()==manifest['source_sha256']
        partitions=json.loads((arm/'partitions.json').read_text())
        config=manifest['config']
        expected=len(config['tasks'].split(','))*len(config['seeds'].split(','))
        done=0
        for task,split in partitions.items():
            sets={s:{p['input'] for p in pairs} for s,pairs in split.items()}
            assert not (sets['test']&sets['dev'] or sets['test']&sets['construction'] or sets['dev']&sets['construction'])
            for file in sorted((arm/task).glob('*/predictions.json')):
                cell=json.loads(file.read_text())
                assert all(p in split['construction'] for p in cell['demos'])
                assert cell['dummy_query'] in sets['construction']
                assert cell['dummy_query'] not in {p['input'] for p in cell['demos']}
                if not cell.get('complete'):continue
                done+=1
                targets=cell['targets'];preds=cell['predictions'];chosen=cell['selected']
                assert json.loads((file.parent/'selection.json').read_text())==chosen
                measured={}
                for variant,L in chosen.items():
                    dev={layer:accuracy(preds[f'dev/{variant}/{layer}'],targets['dev']) for layer in cell['layers']}
                    assert L==min(dev,key=lambda layer:(-dev[layer],layer))
                    measured[variant]=accuracy(preds[f'test/{variant}/{L}'],targets['test'])
                real=cell['method']
                control_names=[v for v in chosen if v!=real]
                assert len(control_names)==(3 if real=='tv' else 2)
                control=float(np.mean([measured[v] for v in control_names]))
                conditional=float(np.mean([accuracy(preds[f'test/{v}/{chosen[real]}'],targets['test']) for v in control_names]))
                zero=accuracy(preds['test/zero/None'],targets['test'])
                icl=accuracy(preds['test/icl/None'],targets['test'])
                gap=icl-zero
                rows.append({'model':cell['model'],'method':real,'task':task,'seed':cell['seed'],
                             'zero':zero,'icl':icl,'accuracy':measured[real],'controls':measured,
                             'control_mean':control,'difference':measured[real]-control,
                             'conditional_control_mean':conditional,'gain_over_zero':measured[real]-zero,
                             'recovery':(measured[real]-zero)/gap if gap>0 else None,'icl_gap':gap})
        coverage.append({'model':config['model'],'method':config['method'],'complete':done,'expected':expected})
    aggregates=[]
    for cov in coverage:
        armrows=[r for r in rows if r['model']==cov['model'] and r['method']==cov['method']]
        by_task={t:[r for r in armrows if r['task']==t] for t in sorted({r['task'] for r in armrows})}
        taskdiff=np.array([np.mean([r['difference'] for r in rs]) for rs in by_task.values()])
        if not len(taskdiff):continue
        rng=np.random.default_rng(20260905)
        draws=rng.choice(taskdiff,size=(10000,len(taskdiff)),replace=True).mean(axis=1)
        aggregates.append(dict(cov,task_count=len(taskdiff),mean_difference=float(taskdiff.mean()),
                               ci95=np.quantile(draws,[.025,.975]).tolist() if len(taskdiff)>=5 else None,
                               task_differences={t:float(np.mean([r['difference'] for r in rs])) for t,rs in by_task.items()},
                               undefined_recovery=sum(r['recovery'] is None for r in armrows)))
    result={'coverage':coverage,'aggregates':aggregates,'cells':rows}
    root.mkdir(parents=True,exist_ok=True)
    (root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Held-out validation status','',
           ('All 96 planned cells are complete and verified. Each arm covers eight tasks and three construction seeds.' if len(coverage)==4 and all(c['complete']==c['expected']==24 for c in coverage) else 'Partial results are descriptive. Task and model coverage was fixed before the run; incomplete arms cannot support the planned full comparison.'), '',
           '| Model | Method | Completed cells | Tasks | Accuracy difference vs independently selected controls | Task-bootstrap 95% interval |',
           '| --- | --- | --- | --- | --- | --- |']
    for c in coverage:
        a=next((a for a in aggregates if a['model']==c['model'] and a['method']==c['method']),None)
        if a:
            interval = f"[{a['ci95'][0]:.4f}, {a['ci95'][1]:.4f}]" if a['ci95'] else 'pending: fewer than five tasks'
            lines.append(f"| {c['model']} | {c['method']} | {c['complete']}/{c['expected']} | {a['task_count']} | {a['mean_difference']:.4f} | {interval} |")
        else:lines.append(f"| {c['model']} | {c['method']} | 0/{c['expected']} | 0 | pending | pending |")
    lines+=['','Intervals based on a few tasks are unstable; seeds reuse test inputs. The primary contrast averages the controls and seeds within tasks, then weights tasks equally. Full cell and task details are in `summary.json`.','',
            'Verification checks split input disjointness, construction-only demonstrations and dummy queries, source-snapshot hashes, frozen development-optimal layers and accuracy recomputed from token predictions.']
    (root/'STATUS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(coverage))
    print(f'Verified {len(rows)} completed cells')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);analyze(p.parse_args().root)
