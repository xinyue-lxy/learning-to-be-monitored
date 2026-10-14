"""CPU query-boundary tests, enabled when dataset/verifier dependencies exist."""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

AVAILABLE=all(importlib.util.find_spec(name) for name in ('numpy','yaml','sympy','pylatexenc'))

@unittest.skipUnless(AVAILABLE,'Optional dataset/verifier dependencies are not installed')
class QueryContracts(unittest.TestCase):
    def test_all_task_loaders_and_query_boundaries(self):
        from monitorability_eval.config import policy,task_config,dataset_manifest
        from monitorability_eval.runner import task_object
        from monitorability_eval.artifacts import rows
        from monitorability_eval.engine.utils import save_jsonl
        from monitorability_eval.engine.pipeline.base import BaseTask
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            td=Path(td)
            for task in policy()['tasks']:
                with self.subTest(task=task):
                    obj=task_object(task,task_config('qwen35_4b',7,8),'0')
                    source=td/task/'task';obj.load_dataset(source)
                    records=rows(source/'shard_1.original,standard.jsonl')
                    self.assertEqual(len(records),2*dataset_manifest()[task]['instances'])
                    for arm in ('original','standard'):
                        row=copy.deepcopy(next(x for x in records if x['metadata']['stress_test']==arm))
                        row['response']=['<reasoning>TEST_COT_SENTINEL</reasoning><answer>TEST_FINAL_SENTINEL</answer>']*8
                        row['verification_result']=[False]*8 if arm=='original' else [True]*4+[False]*4
                        save_jsonl([row],source/f'evaluated_llm_{arm}_n=8.jsonl')
                        save_jsonl([row],source/f'evaluated_llm_{arm}_n=8.tested.jsonl')
                    y=td/task/'y';y.mkdir()
                    BaseTask.prepare_verification_llm_query(obj,source,y)
                    obj.prepare_monitor_llm_query(source,td/task/'z')
                    for row in rows(y/'shard_1.original,standard.jsonl'):
                        prompt=row['messages'][0]['content']
                        self.assertNotIn('TEST_COT_SENTINEL',prompt);self.assertIn('TEST_FINAL_SENTINEL',prompt)
                    for row in rows(td/task/'z/shard_1.original,standard.jsonl'):
                        prompt=row['messages'][0]['content']
                        self.assertIn('TEST_COT_SENTINEL',prompt);self.assertNotIn('TEST_FINAL_SENTINEL',prompt)

    def test_sandbag_wrong_valid_final_only(self):
        from monitorability_eval.engine.pipeline.intervention import BaseSandbagTask
        self.assertFalse(BaseSandbagTask.sandbag_goal_satisfied('A','A'))
        self.assertTrue(BaseSandbagTask.sandbag_goal_satisfied('B','A'))
        self.assertFalse(BaseSandbagTask.sandbag_goal_satisfied('','A'))

    def test_external_math_orm_cannot_run(self):
        from monitorability_eval.engine.pipeline.verification.deepscaler.rewards.math_reward import deepscaler_reward_fn
        self.assertTrue(deepscaler_reward_fn(r'\boxed{2}','2'))
        with self.assertRaises(ValueError):deepscaler_reward_fn(r'\boxed{2}','2',enable_llm=True)

if __name__=='__main__':unittest.main()
