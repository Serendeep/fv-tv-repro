import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fvtv import heldout, tasks


class HeldoutTests(unittest.TestCase):
    def test_all_tasks_disjoint_by_input_and_repeatable(self):
        for name in heldout.TASK_NAMES:
            pairs=tasks.load_task(name)
            split=heldout.partition(pairs)
            self.assertEqual(split,heldout.partition(pairs))
            inputs={s:{p['input'] for p in rows} for s,rows in split.items()}
            self.assertEqual(len(inputs['test']),50)
            self.assertEqual(len(inputs['dev']),25)
            for a,b in [('dev','test'),('construction','test'),('construction','dev')]:
                self.assertFalse(inputs[a]&inputs[b])
            self.assertEqual(set.union(*inputs.values()),{p['input'] for p in pairs})

    def test_duplicates_cannot_cross_partitions(self):
        pairs=[{'input':str(i),'output':str(i)} for i in range(160)]
        pairs += [pairs[0],{'input':'0','output':'alternate'}]
        split=heldout.partition(pairs)
        containing=[rows for rows in split.values() if any(p['input']=='0' for p in rows)]
        self.assertEqual(len(containing),1)
        self.assertEqual(sum(p['input']=='0' for p in containing[0]),2)

    def test_selection_uses_development_accuracy_and_lowest_tie(self):
        self.assertEqual(heldout.select_layer({8:[1,0],4:[1,0],0:[0,0]},[1,1]),4)
        with self.assertRaises(ValueError):
            heldout.select_layer({0:[1]},[1,2])

    def test_scoring(self):
        self.assertEqual(heldout.score([1,2,3],[1,0,3]),2/3)
        with self.assertRaises(ValueError):heldout.score([1],[1,2])


if __name__=='__main__':unittest.main()
