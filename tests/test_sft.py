"""Small CPU tests of the numerical objective, accumulation, masks, and writer."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from methods.sft.common import json_safe, write_json_new
from methods.sft.core import collate_right_padded, cosine_learning_rate, save_checkpoint_atomic, assistant_ce_per_trajectory
from methods.sft.serialization import mask_payload
from methods.sft.train import optimizer_step


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(11,5)

    def forward(self,input_ids,attention_mask,use_cache,return_dict):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=TinyBackbone()
        self.lm_head=nn.Linear(5,11,bias=False)


class TrainingTests(unittest.TestCase):
    def examples(self):
        return [dict(target_path=dict(input_ids=[1,2,3,4],labels=[-100,2,3,-100])),
                dict(target_path=dict(input_ids=[2,4,5,6,7],labels=[-100,-100,5,6,-100])),
                dict(target_path=dict(input_ids=[3,5,6],labels=[-100,5,-100]))]

    def test_ce_matches_independent_full_logits_and_trajectory_weighting(self):
        torch.manual_seed(9)
        model=TinyModel()
        batch=collate_right_padded([r['target_path'] for r in self.examples()],pad_token_id=0,device='cpu')
        actual,counts=assistant_ce_per_trajectory(model,batch)
        logits=model.lm_head(model.model.embedding(batch['input_ids']))[:,:-1,:]
        labels=batch['labels'][:,1:]
        token_loss=torch.nn.functional.cross_entropy(logits.transpose(1,2),labels,ignore_index=-100,reduction='none')
        expected=(token_loss.sum(1)/(labels!=-100).sum(1)).mean()
        self.assertEqual(counts,[2,2,1])
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(batch['attention_mask'][2],torch.tensor([1,1,1,0,0]))
        self.assertEqual(batch['labels'][2,-1].item(),-100)

    def test_partial_batch_accumulation_matches_unsplit_update(self):
        torch.manual_seed(7)
        initial=TinyModel()
        models=[copy.deepcopy(initial) for _ in range(3)]
        losses=[]
        for model,micro in zip(models,[1,2,3]):
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,betas=(.9,.95),eps=1e-8,weight_decay=.01)
            losses.append(optimizer_step(model,optimizer,self.examples(),micro,1.,0,'cpu')['full_assistant_ce'])
        for model in models[1:]:
            for a,b in zip(models[0].parameters(),model.parameters()):
                torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-7)
        self.assertLess(max(losses)-min(losses),1e-6)

    def test_native_mask_includes_payload_only_and_rejects_boundary_drift(self):
        self.assertEqual(mask_payload([1,4],[5,6],[1,4,5,6,2]),[-100,-100,5,6,-100])
        with self.assertRaises(ValueError):mask_payload([1],[3],[1,4,2])
        with self.assertRaises(ValueError):mask_payload([1],[3],[1,3,2],cap=2)

    def test_scheduler_warmup_and_final_floor(self):
        kwargs=dict(total_steps=20,warmup_step_count=2,peak_lr=1e-5,minimum_lr_ratio=.1)
        self.assertEqual(cosine_learning_rate(0,**kwargs),0)
        self.assertAlmostEqual(cosine_learning_rate(1,**kwargs),1e-5/2)
        self.assertEqual(cosine_learning_rate(2,**kwargs),1e-5)
        self.assertAlmostEqual(cosine_learning_rate(20,**kwargs),1e-6)

    def test_loading_report_sets_and_checkpoint_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            write_json_new(path/'report.json',{'loading':{'missing_keys':set()}})
            self.assertEqual(json.loads((path/'report.json').read_text()),{'loading':{'missing_keys':[]}})
            def writer(p):(p/'dummy.txt').write_text('test')
            saved=save_checkpoint_atomic(path/'adapter',adapter_writer=writer,tokenizer_writer=writer,manifest={})
            self.assertFalse(saved['resumable'])
            with self.assertRaises(FileExistsError):
                save_checkpoint_atomic(path/'adapter',adapter_writer=writer,tokenizer_writer=writer,manifest={})





class DataTests(unittest.TestCase):
    def row(self, question='A task'):
        return dict(ordinary_student_messages=[{'role':'system','content':'Answer with reasoning.'},
                    {'role':'user','content':question}],
                    target_assistant_content='<reasoning>Because of the evidence.</reasoning><answer>A</answer>')

    def config(self):
        from methods.sft.common import load_config
        return load_config(Path(__file__).resolve().parents[1]/'examples/sft.yaml')

    def test_external_data_derives_schedule_without_frozen_artifacts(self):
        from methods.sft.common import load_data
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'train.jsonl'
            path.write_text('\n'.join(json.dumps(self.row(f'Task {i}')) for i in range(9)))
            config=self.config()
            train,validation,plans=load_data(config,path)
            self.assertEqual(len(train),9)
            self.assertEqual(validation,[])
            self.assertEqual([len(b) for b in plans[0]['batches']],[8,1])
            self.assertEqual(config['scheduler']['total_optimizer_steps'],2)
            self.assertEqual(config['scheduler']['warmup_steps'],1)
            self.assertEqual(train[0]['target_answer'],'A')

    def test_prompt_and_source_groups_do_not_cross_splits(self):
        from methods.sft.common import load_data
        with tempfile.TemporaryDirectory() as tmp:
            train_path,val_path=Path(tmp)/'train.jsonl',Path(tmp)/'val.jsonl'
            train_path.write_text(json.dumps(self.row()))
            val_path.write_text(json.dumps(self.row()))
            with self.assertRaises(ValueError):load_data(self.config(),train_path,val_path)
            a,b=self.row('A'),self.row('B')
            a['content_group_id']=b['content_group_id']='same-source'
            train_path.write_text(json.dumps(a));val_path.write_text(json.dumps(b))
            with self.assertRaises(ValueError):load_data(self.config(),train_path,val_path)

    def test_epoch_order_obeys_seed_and_keeps_every_sample(self):
        from methods.sft.common import epoch_plan
        rows=[dict(request_id=str(i)) for i in range(20)]
        a,b=epoch_plan(rows,2,7,42),epoch_plan(rows,2,7,43)
        self.assertEqual(a,epoch_plan(rows,2,7,42))
        self.assertNotEqual(a[1]['batches'],b[1]['batches'])
        for plan in a:
            self.assertEqual(sorted(x for batch in plan['batches'] for x in batch),sorted(r['request_id'] for r in rows))

    def test_config_supports_both_architectures(self):
        from methods.sft.common import load_config
        import yaml
        config=self.config()
        config['model_family']='mistral3'
        config['lora'].pop('target_modules')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.yaml';path.write_text(yaml.safe_dump(config))
            self.assertIsInstance(load_config(path)['lora']['target_modules'],str)

    def test_qwen_loader_accepts_sample_configuration_without_private_model_metadata(self):
        from unittest.mock import patch, Mock
        from methods.sft.train import load_model
        model=Mock();model.parameters.return_value=[SimpleNamespace(device=SimpleNamespace(type='cuda'))]
        with patch('methods.sft.train.load_qwen35_text_lora_model',return_value=(model,{})) as loader:
            loaded,_=load_model(self.config(),Path('/example/model'))
            self.assertIs(loaded,model)
            self.assertEqual(loader.call_args.args[0]['model']['snapshot_path'],'/example/model')

    def test_ambiguous_payloads_are_rejected(self):
        from methods.sft.common import validate_row
        row=self.row();row['target_assistant_content']+='<answer>B</answer>'
        with self.assertRaises(ValueError):validate_row(row)



class SerializationTests(unittest.TestCase):
    def serializer(self, max_length=2048):
        from methods.sft.serialization import QwenSerializer
        class Tokenizer:
            def encode(self, text, **kwargs):
                return [ord(c) for c in text]
            def decode(self, ids, **kwargs):
                return ''.join(chr(c) for c in ids)
            def apply_chat_template(self, messages, tokenize, **kwargs):
                text = ''.join(m['content'] for m in messages if m['role'] != 'assistant') + '<assistant>'
                if messages[-1]['role'] == 'assistant':
                    text += messages[-1]['content'] + '<|im_end|>\n'
                return self.encode(text) if tokenize else text
        serializer = QwenSerializer.__new__(QwenSerializer)
        serializer.tokenizer = Tokenizer()
        serializer.max_length = max_length
        return serializer

    def test_payload_masks_without_frozen_token_metadata(self):
        row = DataTests().row()
        row.update(request_id='example',category='cue_answer.math')
        result = self.serializer().serialize(row)['target_path']
        supervised = [token for token in result['labels'] if token != -100]
        self.assertEqual(''.join(chr(c) for c in supervised),row['target_assistant_content'])
        self.assertEqual(result['labels'][-1],-100)
        self.assertFalse(result['truncated'])
        with self.assertRaises(ValueError):
            self.serializer(max_length=4).serialize(row)


if __name__ == '__main__':
    unittest.main()
