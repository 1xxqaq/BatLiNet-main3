"""Synthetic checks for standalone paths, source independence and safe recovery."""
import copy
import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_cross_chem_single_cell as suite
from scripts import run_three_tasks as common
from scripts import transfer_cycle_soft_data as data
from scripts import run_cross_chem_cycle_soft_difference as combo


class SingleCellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self,dropout=.1):
        cfg=copy.deepcopy(suite.read_config())
        cfg['combination']['model'].update(input_height=20,input_width=128,attention_channels=8,
            attention_heads=2,head_hidden_channels=8,encoder_dropout=dropout,attention_dropout=dropout)
        cfg['training'].update(pretrain_epochs=4,target_epochs=4,logical_batch=4,micro_batch=2,inference_chunk=2)
        g=torch.Generator().manual_seed(29)
        source=dict(feature=torch.randn(5,6,20,128,generator=g),label=torch.tensor([250.,340.,470.,650.,800.]),ids=[f'LFP_{i}' for i in range(5)])
        target=dict(feature=torch.randn(3,6,20,128,generator=g),label=torch.tensor([210.,430.,700.]),ids=[f'LCO_{i}' for i in range(3)])
        return cfg,source,target

    def stage(self,tmp,model=suite.DIRECT,n=3):
        return dict(model=model,target='LCO',train_count=n,seed=2,epochs=4,folder=str(Path(tmp)/model))

    def test_frozen_plan_has_224_target_and_8_source_runs(self):
        cfg=suite.read_config();jobs=suite.plan(list(data.TEST_COUNTS));stages=suite.stages('tmp',jobs,cfg)
        self.assertEqual(len(jobs),224);self.assertEqual(len(stages),232)
        self.assertEqual(sum(s['model']==suite.PRETRAIN for s in stages),8)
        self.assertEqual(sum(s['epochs'] for s in stages),23200)
        with self.assertRaises(ValueError):suite.plan(['NCA'],[16])

    def test_identical_own_modules_and_forward_no_relation_parameters(self):
        cfg,source,target=self.fixture()
        a=suite.make_model(cfg,2,'cpu');b=combo.make_model(cfg['combination'],2,'cpu')
        expected={k:v for k,v in b.state_dict().items() if k.startswith(('cell_encoder.','ori_head.'))}
        self.assertEqual(set(a.state_dict()),set(expected))
        for k,v in a.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
        self.assertFalse(hasattr(a,'cross_attention'));self.assertFalse(hasattr(a,'support_head'))
        a.eval();b.eval();x=target['feature']
        torch.testing.assert_close(a(x),b.ori_head(b.cell_encoder(x)),rtol=0,atol=0)
        self.assertTrue(any('cycle_mixer' in k for k in a.state_dict()))

    def test_fixed_transform_handles_one_label_and_round_trip(self):
        cfg,source,target=self.fixture()
        labels=torch.tensor([123.,1000.,3000.])
        z=suite.encode_labels(labels,cfg)
        torch.testing.assert_close(z,(labels.double()/1000).log().float(),rtol=0,atol=0)
        torch.testing.assert_close(suite.decode_labels(z,cfg),labels.double(),rtol=1e-7,atol=1e-5)
        self.assertTrue(torch.isfinite(suite.encode_labels(labels[:1],cfg)).all())
        source['label'].fill_(1e9)
        torch.testing.assert_close(z,suite.encode_labels(labels,cfg),rtol=0,atol=0)
        with self.assertRaises(FloatingPointError):suite.decode_labels(torch.tensor([1000.]),cfg)

    def test_microbatch_weighting_includes_tail(self):
        cfg,source,target=self.fixture(dropout=0.)
        a=suite.make_model(cfg,2,'cpu');b=suite.make_model(cfg,2,'cpu')
        oa=torch.optim.SGD(a.parameters(),lr=.001);ob=torch.optim.SGD(b.parameters(),lr=.001)
        whole=copy.deepcopy(cfg);whole['training']['micro_batch']=4
        x=suite.epoch_train(a,source,cfg,oa,'cpu');y=suite.epoch_train(b,source,whole,ob,'cpu')
        self.assertEqual(x['optimizer_steps'],2)
        self.assertAlmostEqual(x['train_loss'],y['train_loss'],places=5)
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=2e-5,atol=2e-6)

    def test_target_training_never_reads_source_features_or_labels(self):
        cfg,source,target=self.fixture();binding={'method':'direct'}
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b,redirect_stdout(io.StringIO()):
            first=suite.train_stage(target,cfg,self.stage(a),binding,'cpu')
            source['feature'].fill_(1e9);source['label'].fill_(1e9)
            second=suite.train_stage(target,cfg,self.stage(b),binding,'cpu')
            for k in first['state']:torch.testing.assert_close(first['state'][k],second['state'][k],rtol=0,atol=0)

    def test_resume_exact_weights_dropout_rng_and_log_alignment(self):
        cfg,source,target=self.fixture();binding={'case':'resume'}
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b,redirect_stdout(io.StringIO()):
            full=suite.train_stage(target,cfg,self.stage(a),binding,'cpu')
            real_save=common.save
            def interrupt(value,path):
                real_save(value,path)
                if Path(path).name=='progress.pt' and value['epoch']==2:raise RuntimeError('模拟保存第2轮后断电')
            with patch.object(common,'save',side_effect=interrupt):
                with self.assertRaises(RuntimeError):suite.train_stage(target,cfg,self.stage(b),binding,'cpu')
            path=Path(self.stage(b)['folder'])/'train.jsonl'
            with path.open('a',encoding='utf-8') as h:h.write('{"epoch":3,"train_loss":1.0}\n{partial')
            resumed=suite.train_stage(target,cfg,self.stage(b),binding,'cpu')
            for k in full['state']:torch.testing.assert_close(full['state'][k],resumed['state'][k],rtol=0,atol=0)
            self.assertEqual(len(path.read_text().splitlines()),4)
            with self.assertRaises(ValueError):suite.train_stage(target,cfg,self.stage(b),{'different':True},'cpu')

    def test_each_finetune_restarts_same_source_checkpoint(self):
        cfg,source,target=self.fixture()
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            stage=self.stage(tmp,suite.PRETRAIN,len(source['ids']))
            state=suite.train_stage(source,cfg,stage,{'source':'LFP'},'cpu')
            original=copy.deepcopy(state['state']);starts=[];real_epoch=suite.epoch_train
            def observe(model,part,cfg,opt,device):
                if not starts or len(starts)%4==0:
                    snapshot=copy.deepcopy(model.state_dict())
                    for k in original:torch.testing.assert_close(snapshot[k],original[k],rtol=0,atol=0)
                starts.append(True)
                return real_epoch(model,part,cfg,opt,device)
            with patch.object(suite,'epoch_train',side_effect=observe):
                for name in ['first','second']:
                    dest=dict(self.stage(tmp,suite.FINETUNE),folder=str(Path(tmp)/name))
                    suite.train_stage(target,cfg,dest,{'destination':name},'cpu',initial=state['state'])
            for k in original:torch.testing.assert_close(state['state'][k],original[k],rtol=0,atol=0)

    def test_prediction_receipt_and_final_scores_detect_tampering(self):
        cfg,source,target=self.fixture()
        p={'target_train_ids':target['ids'],'target_test_ids':source['ids']}
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            stage=self.stage(tmp);binding=suite.target_binding({'queue':1},stage,p)
            suite.train_stage(target,cfg,stage,binding,'cpu')
            row=suite.evaluate(cfg,stage,binding,p,source,'cpu')
            self.assertEqual(row,suite.saved_result(stage,binding))
            out=common.load(Path(stage['folder'])/'test.pt');out['prediction_cycles'][0]+=1
            common.save(out,Path(stage['folder'])/'test.pt')
            with self.assertRaises(ValueError):suite.saved_result(stage,binding)

    def test_report_recovers_saved_predictions_without_writing_training_files(self):
        cfg,source,target=self.fixture()
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()) as output:
            identity={'config':cfg,'runs':[['LCO',3,2,suite.DIRECT]]}
            stage=suite.stages(tmp,[('LCO',3,2,suite.DIRECT)],cfg)[1]
            p={'target_train_ids':target['ids'],'target_test_ids':source['ids']}
            binding=suite.target_binding(identity,stage,p)
            common.atomic_json(identity,Path(tmp)/'queue.json')
            common.atomic_json({'rows':[]},Path(tmp)/'combination_reference.json')
            common.save(p,Path(tmp)/'protocols/LCO/train_3/seed_2.pt')
            suite.train_stage(target,cfg,stage,binding,'cpu')
            suite.evaluate(cfg,stage,binding,p,source,'cpu')
            before={str(f):common.digest(f) for f in Path(tmp).rglob('*') if f.is_file()}
            suite.report(tmp)
            after={str(f):common.digest(f) for f in Path(tmp).rglob('*') if f.is_file()}
            self.assertEqual(before,after)
            self.assertIn('1/1',output.getvalue())

    def test_eta_remaining_work_and_resume_excludes_downtime(self):
        cfg,source,target=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            schedule=suite.stages(tmp,[('LCO',1,0,suite.DIRECT),('LCO',1,0,suite.FINETUNE)],cfg)
            progress=suite.Progress(tmp,schedule)
            first=progress.publish();self.assertIsNone(first['estimated_end_beijing'])
            progress.sample(275,2.0);progress.sample(1,1.0)
            value=progress.publish(schedule[0],2)
            self.assertEqual(value['estimated_remaining_seconds'],12.)
            self.assertIn('+08:00',value['estimated_end_beijing'])
            final=progress.publish(schedule[1],4);self.assertEqual(final['target_training_finished'],1)

    def test_existing_manifests_and_lists_are_read_only_and_tampering_rejected(self):
        records=[dict(cell_id=f'{chem}_{i:03}',chemistry=chem,label_cycles=200.+i) for chem,n in data.EXPECTED.items() for i in range(n)]
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);policy=dict(data.POLICY,stored_features='raw_unclipped_author_features')
            queue=dict(schema=2,models=[combo.MODEL],data_policy=policy,code_hashes={'old.py':'hash'})
            common.atomic_json(queue,base/'queue.json')
            for r in records:
                path=data.feature_path(base/'dataset',r['cell_id']);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'trusted-fixture')
                r.update(feature_file=path.relative_to(base/'dataset').as_posix(),feature_sha256=common.digest(path))
            manifest=dict(identity=dict(policy=policy,code_hashes=queue['code_hashes']),records=records,observed_counts=data.EXPECTED)
            common.atomic_json(manifest,base/'dataset/manifest.json')
            p,pp=data.protocol(manifest,'LCO',1,0,base,policy)
            before=common.digest(pp)
            _,protocols,receipt=suite.verified_inputs(base,[('LCO',1,0,suite.DIRECT)])
            self.assertEqual(protocols[('LCO',1,0)]['target_train_ids'],p['target_train_ids'])
            self.assertEqual(common.digest(pp),before);self.assertTrue(receipt['reuse_features_without_copy'])
            p['target_train_ids'][0]=p['target_test_ids'][0];common.save(p,pp)
            with self.assertRaises(ValueError):suite.verified_inputs(base,[('LCO',1,0,suite.DIRECT)])

    def test_eight_seed_summary_with_incomplete_group_excluded(self):
        rows=[]
        for seed in range(8):
            for method in (*suite.MODELS,combo.MODEL):
                rows.append(dict(target='LCO',train_count=1,seed=seed,model=method,RMSE=seed+1.,MAE=seed+1.,MAPE=.1+seed*.01,ACC15=.5))
        value=suite.summarize(rows)
        self.assertEqual(len(value['groups']),3);self.assertEqual(len(value['combination_reference_differences']),2)
        self.assertAlmostEqual(value['groups'][0]['metrics']['MAPE']['mean'],.135)
        self.assertEqual(len(suite.summarize(rows[:-1])['groups']),2)


if __name__=='__main__':unittest.main(verbosity=2)
