"""Exercise actual orchestration with fake model calls, including resume."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_heldout as runner


class HeldoutRunnerTests(unittest.TestCase):
    def test_test_calls_follow_all_development_selection_and_resume_is_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=argparse.Namespace(model='gpt-j-6b',method='tv',tasks='antonym',seeds='100',out=tmp)
            model=type('Model',(),{'tokenizer':None})()
            extraction=[];calls=[]
            def extract(m,demos,dummy,remote):
                extraction.append((demos,dummy))
                return {0:torch.tensor([len(extraction)]),3:torch.tensor([len(extraction)])}
            def predictions(m,prompts,*extra,**kwargs):
                calls.append(prompts)
                split=json.loads((Path(tmp)/'partitions.json').read_text())['antonym']
                test_prompts={runner.tasks.build_zeroshot_prompt(p['input']) for p in split['test']}
                if any(p in test_prompts for p in prompts):
                    selection=json.loads((Path(tmp)/'antonym/100/selection.json').read_text())
                    self.assertEqual(set(selection),{'tv','tv_shuffle_0','tv_shuffle_1','tv_shuffle_2'})
                return [1]*len(prompts)
            with patch.object(runner,'load_model',return_value=model),patch.object(runner.fv,'verify_arch'),patch.object(runner.fv,'arch_config',return_value={'n_layers':4}),patch.object(runner.tasks,'target_first_token_id',return_value=1),patch.object(runner.tv,'extract_theta_all_layers',side_effect=extract),patch.object(runner.tv,'patch_theta',side_effect=predictions),patch.object(runner.eval_icl,'predict_top1',side_effect=predictions):
                runner.run(args)
                state=json.loads((Path(tmp)/'antonym/100/predictions.json').read_text())
                self.assertTrue(state['complete'])
                self.assertEqual(len(extraction),4)
                self.assertTrue(all(q==state['dummy_query'] for _,q in extraction))
                self.assertEqual(extraction[0][0],state['demos'])
                for demos,_ in extraction[1:]:
                    self.assertEqual([p['input'] for p in demos],[p['input'] for p in state['demos']])
                    self.assertCountEqual([p['output'] for p in demos],[p['output'] for p in state['demos']])
                n_calls=len(calls)
                runner.run(args)
                self.assertEqual(len(calls),n_calls)
                args.seeds='101'
                with self.assertRaisesRegex(ValueError,'Resume refused'):runner.run(args)

    def test_fv_construction_receives_no_development_or_test_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=argparse.Namespace(model='gpt-j-6b',method='fv',tasks='antonym',seeds='100',out=tmp)
            model=type('Model',(),{'tokenizer':None})()
            def check_pool(m,pairs,*extra,**kwargs):
                split=json.loads((Path(tmp)/'partitions.json').read_text())['antonym']
                self.assertEqual(pairs,split['construction'])
                return torch.zeros(4,2,2)
            def predictions(m,prompts,*extra,**kwargs):return [1]*len(prompts)
            with patch.object(runner,'load_model',return_value=model),patch.object(runner.fv,'verify_arch'),patch.object(runner.fv,'arch_config',return_value={'n_layers':4}),patch.object(runner.tasks,'target_first_token_id',return_value=1),patch.object(runner.fv,'compute_mean_head_activations',side_effect=check_pool) as mean,patch.object(runner.fv,'compute_aie',side_effect=check_pool) as aie,patch.object(runner.fv,'top_k_heads',return_value=[(0,0,1.0)]),patch.object(runner.controls,'pick_random_heads',return_value=[(1,1)]),patch.object(runner.fv,'grab_out_proj_params',return_value={}),patch.object(runner.fv,'compute_fv',return_value=torch.ones(2)),patch.object(runner.fv,'inject_fv',side_effect=predictions),patch.object(runner.eval_icl,'predict_top1',side_effect=predictions):
                runner.run(args)
                self.assertEqual(mean.call_count,1)
                self.assertEqual(aie.call_count,1)
                state=json.loads((Path(tmp)/'antonym/100/predictions.json').read_text())
                self.assertTrue(state['complete'])
                self.assertEqual(set(state['selected']),{'fv','fv_random_heads','fv_random_vector'})


if __name__=='__main__':unittest.main()
