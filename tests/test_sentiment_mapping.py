import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_sentiment_mapping as runner

class MappingTest(unittest.TestCase):
    def rows(self):
        return [{'input':f'{label}-{i}','output':label} for label in ['positive','negative'] for i in range(80)]
    def test_disjoint_balanced_and_conflicts(self):
        rows=self.rows()
        rows.extend([rows[0],dict(input=rows[1]['input'],output='negative')])
        split,audit=runner.partition(rows)
        self.assertEqual(audit['retained_inputs'],159)
        self.assertEqual(len(audit['conflicting_inputs_excluded']),1)
        groups=[{p['input'] for p in v} for v in split.values()]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        for section,n in [('dev',20),('test',40)]:
            self.assertEqual(sum(p['output']=='positive' for p in split[section]),n)
            self.assertEqual(sum(p['output']=='negative' for p in split[section]),n)
        for seed in runner.SEEDS:
            demos=runner.demonstrations(split['construction'],seed)
            self.assertEqual(sum(p['output']=='positive' for p in demos),5)
            ab,ba=runner.mapped(demos,'ab'),runner.mapped(demos,'ba')
            self.assertEqual([p['input'] for p in ab],[p['input'] for p in ba])
            self.assertTrue(all(p['output']!=q['output'] for p,q in zip(ab,ba)))
    def test_pilot_isolation_and_full_resume(self):
        class Tokenizer:
            def encode(self,s,**kwargs):
                for label,index in [('positive',1),('negative',2),('A',3),('B',4)]:
                    if s==' '+label: return [index]
                    if s.endswith(' '+label): return [0,index]
                return [0]
        model=SimpleNamespace(tokenizer=Tokenizer(),config=SimpleNamespace(max_position_embeddings=2048))
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            data=base/'data.json'; data.write_text(json.dumps(self.rows()))
            args=SimpleNamespace(data=str(data),out=str(base/'out'),model='gpt-j-6b',phase='pilot')
            def predictions(model,prompts,*a,**kw):
                if len(prompts)==80:
                    # Before any test call all four selection choices must be on disk.
                    selections=list((base/'out').glob('*/*/selection.json'))
                    self.assertTrue(selections)
                    self.assertTrue(all(len(json.loads(p.read_text()))==4 for p in selections))
                return [1]*len(prompts)
            with patch.object(runner,'load_model',return_value=model), patch.object(runner.fv,'verify_arch'), patch.object(runner.fv,'arch_config',return_value={'n_layers':2}), patch.object(runner.tv,'extract_theta_all_layers',return_value={0:runner.torch.zeros(2),1:runner.torch.zeros(2)}), patch.object(runner.tv,'patch_theta',side_effect=predictions), patch.object(runner.eval_icl,'predict_top1',side_effect=predictions):
                runner.run(args)
                states=[json.loads(p.read_text()) for p in (base/'out').glob('*/*/predictions.json')]
                self.assertEqual(len(states),3)
                self.assertTrue(all(s['pilot_complete'] and all(k.startswith('dev/') for k in s['predictions']) for s in states))
                args.phase='full'; runner.run(args)
                states=[json.loads(p.read_text()) for p in (base/'out').glob('*/*/predictions.json')]
                self.assertEqual(len(states),9)
                self.assertTrue(all(s['complete'] for s in states))
                data.write_text('[]')
                with self.assertRaisesRegex(ValueError,'Resume refused'): runner.run(args)
if __name__=='__main__': unittest.main()
