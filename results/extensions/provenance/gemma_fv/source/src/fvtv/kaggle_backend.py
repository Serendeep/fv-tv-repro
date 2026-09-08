"""FP16 native-hook backend validated against Gemma NDIF development outputs."""
import gc,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from . import tasks
CACHE={}
REVISIONS={}
TOKEN=None
DEADLINE=float('inf')
OUTPUT=None

def load_model(hf_id, remote=True):
    if CACHE.get('id')==hf_id:return CACHE['model']
    CACHE.clear();gc.collect();torch.cuda.empty_cache()
    tok=AutoTokenizer.from_pretrained(hf_id,revision=REVISIONS[hf_id],token=TOKEN)
    tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    hf=AutoModelForCausalLM.from_pretrained(hf_id,revision=REVISIONS[hf_id],token=TOKEN,torch_dtype=torch.float16,device_map='auto',max_memory={0:'13GiB',1:'13GiB'},attn_implementation='eager',low_cpu_mem_usage=True)
    hf.eval();hf.config.use_cache=False
    assert all(str(v) not in ['cpu','disk'] for v in hf.hf_device_map.values())
    m=SimpleNamespace(hf=hf,config=hf.config,tokenizer=tok)
    CACHE.update(id=hf_id,model=m)
    if OUTPUT:
        (OUTPUT/(hf_id.split('/')[-1]+'-device-map.json')).write_text(json.dumps({k:str(v) for k,v in hf.hf_device_map.items()},indent=2))
    return m

def forward(model,prompts):
    if time.time()>DEADLINE:raise TimeoutError('Fixed experiment deadline reached')
    enc=model.tokenizer(prompts,return_tensors='pt',padding=True).to(model.hf.get_input_embeddings().weight.device)
    with torch.inference_mode():logits=model.hf(**enc).logits[:,-1,:].float().cpu()
    if not torch.isfinite(logits).all():raise ValueError('Non-finite FP16 logits')
    return logits

def extract_theta_all_layers(model,demos,dummy_query,remote=True,prompt=None):
    prompt=prompt or tasks.build_icl_prompt(demos,dummy_query)
    vectors={};handles=[]
    def hook(L):
        def capture(module,args,output):
            h=output[0] if isinstance(output,tuple) else output
            vectors[L]=h[0,-1,:].detach().float().cpu().clone()
        return capture
    try:
        for L,block in enumerate(model.hf.model.layers):handles.append(block.register_forward_hook(hook(L)))
        forward(model,[prompt])
    finally:
        for h in handles:h.remove()
    if not all(torch.isfinite(v).all() and v.norm()>0 for v in vectors.values()):raise ValueError('Invalid vector')
    return vectors

def predict_top1(model,prompts,batch_size=4,remote=True):
    return [p for i in range(0,len(prompts),batch_size) for p in forward(model,prompts[i:i+batch_size]).argmax(-1).tolist()]

def patch_theta(model,prompts,theta,layer,batch_size=4,remote=True,additive=False):
    def hook(module,args,output):
        h=output[0] if isinstance(output,tuple) else output
        h=h.clone()
        if additive:h[:,-1,:]=h[:,-1,:]+theta.to(h)
        else:h[:,-1,:]=theta.to(h)
        return (h,)+output[1:] if isinstance(output,tuple) else h
    handle=model.hf.model.layers[layer].register_forward_hook(hook)
    try:return predict_top1(model,prompts,batch_size,remote)
    finally:handle.remove()

def verify_arch(model,remote=True):
    cfg=model.config
    assert cfg.model_type=='gemma2' and cfg.num_hidden_layers==42 and cfg.hidden_size==3584 and cfg.head_dim==256
