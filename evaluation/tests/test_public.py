"""Synthetic CPU tests only; no model outputs or experiment results."""
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

from monitorability_eval import artifacts
from monitorability_eval.analysis import compare
from monitorability_eval.bounds import recall_bounds,instance_lower_metrics,summarize_task
from monitorability_eval.cli import main
from monitorability_eval.config import verify_assets,policy,profiles,task_config,monitor_config
from monitorability_eval.lora_mapping import convert,equal_payloads
from monitorability_eval.runner import plan_task,run_task
from monitorability_eval.scoring import compute_input_intervention_metrics,parse_monitor_verdict,InputInterventionMetricError

def instance(i,q=.5,z=.5,tnr=.8):
    return dict(task_id=i,r_effect=q,p_z1_x1_y1=z,TPR_mc_i=min(1,z/q),TNR_X1_i=tnr,
        modified_gmean2_i=min(1,z/q)*tnr,metric_valid=True,effect_eligible=True,tnr_estimable=True,
        n_X1_Y0_valid_monitor_task_rollouts=2)

def cell(label,ids,seed=7,task='cue_answer.health'):
    rows=[instance(i) for i in ids]
    return dict(identity=dict(label=label,seed=seed,task=task,model_key='toy',model_revision='toy-revision',
        monitor='toy-monitor',monitor_revision='toy-revision',policy_fingerprint='toy-policy',comparison_fingerprint=f'{task}:{seed}'),
        per_instance=rows,usable=len(rows),effect_eligible=len(rows),estimable=len(rows),na_reason=None if rows else 'no_usable_two_arm_instances',diagnostics={})

def compare_toy(cells,**kw):
    opts=dict(base='base',candidates=['candidate'],seeds=[7],support='native',tasks=['cue_answer.health'])
    opts.update(kw);return compare(cells,**opts)

class BoundsTests(unittest.TestCase):
    def test_endpoints(self):
        lo,hi=recall_bounds(.5,.75);self.assertEqual((lo,hi),(.5,1))
        self.assertEqual(recall_bounds(1,.25),(.25,.25))
    def test_invalid_components(self):
        for q,z in [(0,.5),(-1,.5),(True,.5),(.5,float('nan'))]:
            with self.assertRaises(ValueError):recall_bounds(q,z)
    def test_keep_saved_na(self):
        self.assertIsNone(instance_lower_metrics({'modified_gmean2_i':None}))
    def test_do_not_invent_missing_bound_components(self):
        row=instance('x');del row['r_effect']
        with self.assertRaises(ValueError):instance_lower_metrics(row)
    def test_preserve_support_gate(self):
        row=instance('x');row['n_X1_Y0_valid_monitor_task_rollouts']=1
        with self.assertRaises(ValueError):instance_lower_metrics(row)
    def test_reject_changed_upper(self):
        row=instance('x');row['TPR_mc_i']=.1
        with self.assertRaises(ValueError):instance_lower_metrics(row)
    def test_products_before_means(self):
        s=summarize_task([instance('a',1,.2,1),instance('b',1,1,.2)],task_reportable=True)['values']
        self.assertAlmostEqual(s['upper_score'],.2)
        self.assertNotAlmostEqual(s['upper_score'],s['upper']*s['TNR_X1'])
    def test_duplicates(self):
        with self.assertRaises(ValueError):summarize_task([instance('x'),instance('x')],task_reportable=True)

class SupportTests(unittest.TestCase):
    def test_native_and_common(self):
        cells=[cell('base',['a','b']),cell('candidate',['b','c'])]
        n=compare_toy(cells)['comparisons']['candidate']['per_seed']['7']['cue_answer.health']
        c=compare_toy(cells,support='common')['comparisons']['candidate']['per_seed']['7']['cue_answer.health']
        self.assertEqual(n['base']['estimable'],2);self.assertEqual(n['candidate']['estimable'],2)
        self.assertEqual(c['common_count'],1);self.assertEqual(c['base']['instance_ids'],['b'])
    def test_trajectory_intersection(self):
        cells=[cell('base',['a','b','c']),cell('candidate',['a','b']),cell('second',['b','c'])]
        report=compare_toy(cells,candidates=['candidate','second'],support='trajectory_common')
        for value in report['comparisons'].values():
            self.assertEqual(value['per_seed']['7']['cue_answer.health']['base']['instance_ids'],['b'])
    def test_empty_common_is_na_not_zero(self):
        result=compare_toy([cell('base',['a']),cell('candidate',['b'])],support='common')
        c=result['comparisons']['candidate']['per_seed']['7']['cue_answer.health']
        self.assertIsNone(c['base']['values']);self.assertEqual(c['base']['na_reason'],'zero_common_estimable')
    def test_strict_seed_task_na_and_valid_zero(self):
        tasks=['cue_answer.health','cue_answer.math'];cells=[]
        for label in ['base','candidate']:
            for seed in [7,8]:
                a=cell(label,['a'],seed,tasks[0]);a['per_instance'][0]=instance('a',z=0)
                cells.extend([a,cell(label,[] if seed==8 else ['b'],seed,tasks[1])])
        r=compare_toy(cells,seeds=[7,8],tasks=tasks)['comparisons']['candidate']
        self.assertIsNone(r['per_task'][tasks[1]]['base']['upper_score']['mean'])
        self.assertEqual(r['macro']['base']['task_count'],1)
        self.assertEqual(r['macro']['base']['values']['upper_score']['mean'],0)
    def test_missing_seed_rejected(self):
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            compare_toy([cell('base',['a']),cell('candidate',['a'])],seeds=[7,8])
    def test_incompatible_provenance(self):
        cells=[cell('base',['a']),cell('candidate',['a'])];cells[1]['identity']['comparison_fingerprint']='different'
        with self.assertRaisesRegex(ValueError,'provenance'):compare_toy(cells)
    def test_no_monitor_pooling(self):
        cells=[cell('base',['a']),cell('candidate',['a'])];cells[1]['identity']['monitor']='another'
        with self.assertRaisesRegex(ValueError,'pool'):compare_toy(cells)
    def test_canary_is_not_formal(self):
        cells=[cell('base',['a']),cell('candidate',['a'])];cells[1]['diagnostic_only']=True
        with self.assertRaisesRegex(ValueError,'canary'):compare_toy(cells)

class VerdictTests(unittest.TestCase):
    def test_parser(self):
        self.assertTrue(parse_monitor_verdict('<answer>A</answer>',allowed_choices='AB'))
        self.assertFalse(parse_monitor_verdict('B',allowed_choices='AB'))
        for text in ['I choose A','AB','<answer>C</answer>','<answer>A</answer><answer>B</answer>']:
            with self.assertRaises(InputInterventionMetricError):parse_monitor_verdict(text,allowed_choices='AB')
    def score(self,responses,labels):
        return compute_input_intervention_metrics([dict(task_id='toy_standard_cot',response=responses,
            verification_result=labels,metadata=dict(r_effect=.5,effect_eligible=True))],
            monitor_scope='cot',monitor_rollouts=4,drop_invalid_monitor_verdicts=True,monitor_allowed_choices='AB')
    def test_invalid_dropped_not_zero(self):
        r=self.score(['A','A','bad','A']+['B']*8,[True,False,False])
        self.assertEqual(r['monitor_parse_dropped_count'],1)
        self.assertEqual(r['per_instance'][0]['p_z1_x1_y1'],1)
        self.assertEqual(r['per_instance'][0]['modified_gmean2_i'],1)
    def test_monitor_repeats_not_distinct_task_rollouts(self):
        r=self.score(['A']*4+['B']*4,[True,False])
        self.assertIsNone(r['per_instance'][0]['modified_gmean2_i'])
    def test_all_invalid_negative_loses_support(self):
        r=self.score(['A']*4+['B']*4+['bad']*4,[True,False,False])
        self.assertIsNone(r['per_instance'][0]['modified_gmean2_i'])

class PlumbingTests(unittest.TestCase):
    def test_assets(self):
        self.assertEqual(verify_assets()['datasets'],12)
    def test_profile_identity_and_explicit_seed(self):
        self.assertEqual(task_config('qwen35_4b',7,8)['evaluated_llm']['chat_template_kwargs'],{'enable_thinking':False})
        self.assertEqual(task_config('mistral_small32_24b',7,8)['evaluated_llm']['chat_template_kwargs'],{})
        for name in profiles()['monitors']:
            c=monitor_config(name,7,'cue_answer.health')
            self.assertEqual(c['rollout_number'],4);self.assertFalse(c['forced_final_choice']);self.assertEqual(c['forced_final_choices'],'AB')
        with self.assertRaises(ValueError):task_config('qwen35_4b',None,8)
    def test_dry_run_no_gpu_or_file_writes(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'not-created';before=dict(os.environ)
            plan=plan_task('qwen35_4b','cue_answer.health',7,8,'base')
            self.assertFalse(run_task(plan,out)['executed']);self.assertFalse(out.exists())
            self.assertEqual(before,dict(os.environ));self.assertNotIn('torch',sys.modules)
    def test_cli_requires_seed(self):
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            main(['task','--model','qwen35_4b','--task','cue_answer.health','--label','base','--output','unused'])
    def test_bundle_immutability_and_exclusive_write(self):
        with tempfile.TemporaryDirectory() as td:
            root=artifacts.new_directory(Path(td)/'bundle');artifacts.write(root/'data.json',{'toy':True})
            artifacts.seal(root);artifacts.verify(root)
            with self.assertRaises(FileExistsError):artifacts.write(root/'data.json',{})
            (root/'data.json').write_text('{}')
            with self.assertRaises(ValueError):artifacts.verify(root)
    def test_lossless_lora_key_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/'source';src.mkdir();h={};off=0
            for side,shape in [('A',[32,2]),('B',[2,32])]:
                key=f'base_model.model.model.layers.0.self_attn.q_proj.lora_{side}.weight'
                h[key]={'dtype':'F32','shape':shape,'data_offsets':[off,off+256]};off+=256
            raw=json.dumps(h).encode();payload=bytes(range(256))*2
            (src/'adapter_model.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw+payload)
            (src/'adapter_config.json').write_text('{}')
            dst=Path(td)/'derived';report=convert(src,dst,expected_count=2)
            self.assertFalse(report['tensor_values_changed'])
            self.assertEqual(report['tensor_payload_sha256'],equal_payloads(src/'adapter_model.safetensors',dst/'adapter_model.safetensors'))

if __name__=='__main__':unittest.main()
