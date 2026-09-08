"""Native FV port: pre-o_proj head means/AIE, projected FV, block-output addition."""
import time
import numpy as np
import torch
from . import tasks,fv,kaggle_backend as backend


def projections(model):return [block.self_attn.o_proj for block in model.hf.model.layers]

def compute_mean_head_activations(model,task_pairs,seed,n_ex=32,n_shot=10,remote=True,batch_size=4):
    cfg=fv.arch_config(model);nh,hd=cfg['n_heads'],cfg['head_dim']
    pool,_=tasks.split_pairs(task_pairs,seed,n_eval=50)
    rng=np.random.default_rng(seed);prompts=[]
    for _ in range(n_ex):
        ts=int(rng.integers(0,2**31-1));demos=tasks.sample_demos(pool,ts,n_shot=n_shot)
        query=pool[int(rng.integers(0,len(pool)))]['input']
        prompts.append(tasks.build_icl_prompt(demos,query))
    total=torch.zeros(cfg['n_layers'],nh,hd)
    for start in range(0,len(prompts),batch_size):
        captured={};handles=[]
        def hook(L):
            def read(module,args):
                x=args[0][:,-1,:]
                assert x.shape[-1]==nh*hd
                captured[L]=x.detach().float().reshape(-1,nh,hd).sum(0).cpu()
            return read
        try:
            for L,proj in enumerate(projections(model)):handles.append(proj.register_forward_pre_hook(hook(L)))
            backend.forward(model,prompts[start:start+batch_size])
        finally:
            for h in handles:h.remove()
        total+=torch.stack([captured[L] for L in range(cfg['n_layers'])])
    result=total/n_ex
    assert torch.isfinite(result).all()
    return result


def patched_logits(model,prompt,means,head_pairs):
    """One row per (layer,head), with exactly that row's head replaced."""
    cfg=fv.arch_config(model);nh,hd=cfg['n_heads'],cfg['head_dim'];handles=[]
    groups={L:[(row,H) for row,(l,H) in enumerate(head_pairs) if l==L] for L,H in head_pairs}
    def hook(rows):
        def patch(module,args):
            x=args[0].clone()
            for row,H,L in rows:x[row,-1,H*hd:(H+1)*hd]=means[L,H].to(x)
            return (x,)+args[1:]
        return patch
    try:
        for L,rows in groups.items():
            handles.append(projections(model)[L].register_forward_pre_hook(hook([(row,H,L) for row,H in rows])))
        return backend.forward(model,[prompt]*len(head_pairs))
    finally:
        for h in handles:h.remove()


def compute_aie(model,task_pairs,mean_activations,seed,n_trials=10,n_shot=10,remote=True,layer_chunk=None,max_batch_rows=4):
    cfg=fv.arch_config(model);pool,queries=tasks.split_pairs(task_pairs,seed,n_eval=50)
    rng=np.random.default_rng(seed+999);result=torch.zeros(n_trials,cfg['n_layers'],cfg['n_heads'])
    # Four patched rows fit the validated T4 memory envelope, regardless of caller cap.
    batch=min(4,max_batch_rows)
    heads=[(L,H) for L in range(cfg['n_layers']) for H in range(cfg['n_heads'])]
    for trial in range(n_trials):
        ts=int(rng.integers(0,2**31-1));demos=tasks.sample_demos(pool,ts,n_shot=n_shot)
        query=queries[int(rng.integers(0,len(queries)))]
        prompt=tasks.build_icl_prompt(demos,query['input'],shuffle_labels=True,shuffle_seed=ts)
        target=tasks.target_first_token_id(model.tokenizer,query['output'])
        clean=backend.forward(model,[prompt])[0].softmax(-1)[target]
        for start in range(0,len(heads),batch):
            pairs=heads[start:start+batch]
            probs=patched_logits(model,prompt,mean_activations,pairs).softmax(-1)[:,target]
            for (L,H),prob in zip(pairs,probs):result[trial,L,H]=prob-clean
        print('AIE trial',trial+1,'/',n_trials,'complete',flush=True)
    assert torch.isfinite(result).all()
    return result.mean(0)


def grab_out_proj_params(model,cfg,layers=None,remote=True):
    if layers is None:layers=range(cfg['n_layers'])
    return {L:(projections(model)[L].weight.detach(),False) for L in layers}


def inject_fv(model,prompts,fv_vector,layer,batch_size=4,remote=True):
    return backend.patch_theta(model,prompts,fv_vector,layer,batch_size=batch_size,additive=True)


def engineering_pilot(model,construction):
    started=time.time();cfg=fv.arch_config(model)
    means=compute_mean_head_activations(model,construction,100,n_ex=4)
    demos=tasks.sample_demos(construction,100)
    query=next(p['input'] for p in construction if p not in demos)
    prompt=tasks.build_icl_prompt(demos,query)
    pairs=[(0,0),(0,15),(20,3),(41,15)]
    batched=patched_logits(model,prompt,means,pairs)
    scalar=torch.cat([patched_logits(model,prompt,means,[pair]) for pair in pairs])
    maxdiff=float((batched-scalar).abs().max())
    assert torch.allclose(batched,scalar,atol=.05,rtol=.005), 'Head batching differs materially from scalar patches'
    assert torch.equal(batched.argmax(-1),scalar.argmax(-1)), 'Head batching changed top-1'
    clean=backend.predict_top1(model,[prompt])
    for L in [0,20,41]:assert inject_fv(model,[prompt],torch.zeros(cfg['hidden']),L)==clean
    # Assert additive writes at both GPU partitions, independently of top-1 changes.
    writes=[]
    for L in [0,20,41]:
        block=model.hf.model.layers[L];captured={}
        def before(module,args,output):
            h=output[0] if isinstance(output,tuple) else output
            captured['before']=h[:,-1,:].detach().clone()
        def add(module,args,output):
            h=output[0] if isinstance(output,tuple) else output;h=h.clone();h[:,-1,:]+=1
            return (h,)+output[1:] if isinstance(output,tuple) else h
        def after(module,args,output):
            h=output[0] if isinstance(output,tuple) else output
            assert torch.equal(h[:,-1,:],captured['before']+1)
        hs=[block.register_forward_hook(fn) for fn in [before,add,after]]
        try:backend.forward(model,[prompt])
        finally:
            for h in hs:h.remove()
        writes.append(L)
    # Validate all heads sum to the full projection in FP32 (geometry, including GQA).
    proj=projections(model)[20];x=means[20].flatten().to(proj.weight.device)
    w=proj.weight.detach().float();parts=[]
    for H in range(cfg['n_heads']):
        hd=cfg['head_dim'];parts.append(w[:,H*hd:(H+1)*hd]@x[H*hd:(H+1)*hd])
    expected=w@x
    assert torch.allclose(sum(parts),expected,atol=1e-4,rtol=1e-4), 'Head projection decomposition mismatch'
    # One full construction-only AIE trial provides a runtime estimate, not a task verdict.
    t=time.time();aie=compute_aie(model,construction,means,100,n_trials=1)
    return {'passed':True,'seconds':time.time()-started,'aie_trial_seconds':time.time()-t,'head_batch_max_logit_difference':maxdiff,'additive_write_layers':writes,'aie_min':float(aie.min()),'aie_max':float(aie.max())}
