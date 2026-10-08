"""Synthetic protocol, pair training, inference and restart checks; no real training."""
import copy
import io
import json
import pickle
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_cross_chem_cycle_soft_difference as suite
from scripts import transfer_cycle_soft_data as data


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def records(self):
        return [dict(cell_id=f'{chem}_{i:03}',chemistry=chem,label_cycles=200.+i)
                for chem,n in data.EXPECTED.items() for i in range(n)]

    def fixture(self,dropout=0.):
        cfg=copy.deepcopy(suite.read_config())
        cfg['model'].update(input_height=20,input_width=128,attention_channels=8,
            attention_heads=2,head_hidden_channels=8,encoder_dropout=dropout,attention_dropout=dropout)
        cfg['runner'].update(maximum_epochs=4,patience=10,logical_batch=4,micro_batch=2,inference_chunk=2)
        rng=torch.Generator().manual_seed(91)
        parts=[]
        for name,n in [('source',3),('train',2),('test',2)]:
            parts.append(dict(feature=torch.randn(n,6,20,128,generator=rng),
                label=torch.linspace(230,1000,n),ids=[f'{name}_{i}' for i in range(n)]))
        source,target,test=parts;mean,scale=data.label_statistics(source,target)
        p=dict(seed=2,target_chemistry='LCO',target_train_count=2,source_ids=source['ids'],
            target_train_ids=target['ids'],target_test_ids=test['ids'],
            train_pairs=torch.tensor([4,1,5,0,3]),val_pairs=torch.tensor([2]))
        return cfg,suite.add_z(source,mean,scale),suite.add_z(target,mean,scale),test,p,mean,scale

    def test_frozen_configuration_and_complete_112_run_plan(self):
        self.assertEqual(len(suite.plan(['LCO','NCA','NMC'],None)),112)
        self.assertEqual(len(suite.plan(['LCO'],[1,16])),16)
        with self.assertRaises(ValueError):suite.plan(['NCA'],[16])
        with self.assertRaises(ValueError):suite.plan(['NCA','NCA'],[1])
        cfg=suite.read_config();m=suite.make_model(cfg,0,'cpu')
        self.assertEqual(sum(p.numel() for p in m.parameters()),211556)
        self.assertEqual(m.cell_encoder.num_tokens,775)

    def test_fixed_tests_nested_training_lfp_only_and_pair_partition(self):
        records=self.records()
        for chem in data.TRAIN_COUNTS:
            for seed in range(8):
                previous=[];test_ids=None
                for n in data.TRAIN_COUNTS[chem]:
                    p=data.build_protocol(records,chem,n,seed);data.validate_protocol(p,records)
                    self.assertEqual(len(p['source_ids']),275)
                    self.assertTrue(all(i.startswith('LFP_') for i in p['source_ids']))
                    self.assertEqual(len(p['target_test_ids']),data.TEST_COUNTS[chem])
                    self.assertEqual(p['target_train_ids'][:len(previous)],previous)
                    if test_ids is not None:self.assertEqual(p['target_test_ids'],test_ids)
                    self.assertFalse(set(p['train_pairs'].tolist())&set(p['val_pairs'].tolist()))
                    previous=p['target_train_ids'];test_ids=p['target_test_ids']
        corrupt=data.build_protocol(records,'LCO',1,0)
        corrupt['train_pairs'][0]=corrupt['val_pairs'][0]
        with self.assertRaises(ValueError):data.validate_protocol(corrupt,records)
        with self.assertRaises(ValueError):data.validate_counts(records[:-1])

    def test_label_statistics_exclude_test_and_use_sample_std(self):
        cfg,source,target,test,p,mean,scale=self.fixture()
        logs=torch.cat([source['label'],target['label']]).double().log()
        torch.testing.assert_close(mean,logs.mean().float(),rtol=0,atol=0)
        torch.testing.assert_close(scale,logs.std(unbiased=True).float(),rtol=0,atol=0)
        test['label'].fill_(1e9)
        a,b=data.label_statistics(source,target)
        torch.testing.assert_close(a,mean,rtol=0,atol=0)
        torch.testing.assert_close(b,scale,rtol=0,atol=0)

    def test_weighted_microbatches_include_partial_logical_batch(self):
        cfg,source,target,test,p,mean,scale=self.fixture()
        a=suite.make_model(cfg,2,'cpu');b=suite.make_model(cfg,2,'cpu')
        oa=torch.optim.SGD(a.parameters(),lr=.001);ob=torch.optim.SGD(b.parameters(),lr=.001)
        whole=copy.deepcopy(cfg);whole['runner']['micro_batch']=4
        x=suite.train_epoch(a,oa,source,target,p['train_pairs'],cfg,'cpu')
        y=suite.train_epoch(b,ob,source,target,p['train_pairs'],whole,'cpu')
        self.assertEqual(x['optimizer_steps'],2)
        self.assertAlmostEqual(x['train_loss'],y['train_loss'],places=6)
        for key,value in a.state_dict().items():
            torch.testing.assert_close(value,b.state_dict()[key],rtol=1e-5,atol=2e-7)

    def test_cached_all_reference_inference_matches_direct_pair_forward(self):
        cfg,source,target,test,p,mean,scale=self.fixture()
        model=suite.make_model(cfg,2,'cpu').eval()
        a=suite.predict(model,source,test,mean,scale,1,'cpu')
        b=suite.predict(model,source,test,mean,scale,3,'cpu')
        with torch.no_grad():
            own,sup,agg,_,_=model.compute_prediction_components(test['feature'],
                source['feature'][None].expand(2,-1,-1,-1,-1).contiguous(),source['z'][None].expand(2,-1))
        for result in [a,b]:
            torch.testing.assert_close(result['diagnostics']['y_ori'],own,rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(result['diagnostics']['y_sup'],sup,rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(result['prediction_cycles'],
                (model.combine_predictions(own,agg)*scale+mean).exp(),rtol=1e-5,atol=1e-4)
            self.assertEqual(result['diagnostics']['support_index'].tolist(),[[0,1,2],[0,1,2]])

    def test_early_stopping_saves_independent_best_state_and_no_test_needed(self):
        cfg,source,target,test,p,mean,scale=self.fixture()
        cfg['runner'].update(patience=2,maximum_epochs=6)
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            folder=Path(tmp)/'LCO/train_2/run'
            with patch.object(suite,'validation_loss',side_effect=[1.,.5,.6,.7]):
                state=suite.train_run(source,target,p,cfg,folder,{'test':'early'},'cpu')
            self.assertEqual(state['trained_epochs'],4);self.assertEqual(state['selected_epoch'],2)
            progress=suite.common.load(folder/'progress.pt')
            self.assertTrue(any(not torch.equal(state['state'][key],progress['state'][key])
                for key in state['state']))
            for key,value in state['state'].items():
                torch.testing.assert_close(value,progress['best_state'][key],rtol=0,atol=0)
            self.assertFalse((folder/'test.pt').exists())

    def test_crash_resume_exact_weights_rng_and_log_reconciliation(self):
        cfg,source,target,test,p,mean,scale=self.fixture(dropout=.1)
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);complete=root/'a/LCO/train_2/run';restart=root/'b/LCO/train_2/run'
            binding={'test':'resume'}
            a=suite.train_run(source,target,p,cfg,complete,binding,'cpu')
            original=suite.common.save
            def interrupt(value,path):
                original(value,path)
                if Path(path).name=='progress.pt' and value['epoch']==2:
                    raise RuntimeError('synthetic interruption after durable epoch 2')
            with patch.object(suite.common,'save',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):
                    suite.train_run(source,target,p,cfg,restart,binding,'cpu')
            # Simulate log written for a partial next epoch; recovery must remove it.
            with (restart/'train.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(epoch=3,train_loss=999.))+'\n')
            b=suite.train_run(source,target,p,cfg,restart,binding,'cpu')
            for key,value in a['state'].items():
                torch.testing.assert_close(value,b['state'][key],rtol=0,atol=0)
            self.assertEqual(a['selected_epoch'],b['selected_epoch'])
            self.assertEqual(a['validation_relation_mse'],b['validation_relation_mse'])
            aa=[json.loads(line) for line in (complete/'train.jsonl').read_text().splitlines()]
            bb=[json.loads(line) for line in (restart/'train.jsonl').read_text().splitlines()]
            self.assertEqual([r['epoch'] for r in bb],[1,2,3,4])
            self.assertEqual([r['train_loss'] for r in aa],[r['train_loss'] for r in bb])
            with self.assertRaises(ValueError):suite.train_run(source,target,p,cfg,restart,{'test':'changed'},'cpu')

    def test_predictions_metrics_completed_receipts_and_tamper_detection(self):
        cfg,source,target,test,p,mean,scale=self.fixture()
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);folder=root/'LCO/train_2'/f'{suite.MODEL}_seed2'
            identity=dict(model=suite.MODEL,runs=[['LCO',2,2]])
            manifest=root/'dataset/manifest.json';protocol=root/'protocols/LCO/train_2/seed_2.pt'
            suite.common.atomic_json({'test':'manifest'},manifest);suite.common.save(p,protocol)
            binding=dict(queue=identity,target='LCO',train_count=2,seed=2,
                manifest_sha256=suite.common.digest(manifest),protocol_sha256=suite.common.digest(protocol))
            suite.train_run(source,target,p,cfg,folder,binding,'cpu')
            row=suite.evaluate(source,test,p,cfg,folder,binding,mean,scale,'cpu')
            again=suite.evaluate(source,test,p,cfg,folder,binding,mean,scale,'cpu')
            self.assertEqual(row,again)
            self.assertEqual(suite.completed_rows(root,identity),[row])
            suite.common.atomic_json(identity,root/'queue.json')
            suite.report(root)
            self.assertFalse((root/'summary.json').exists(), '查看结果不得与训练队列同时写汇总')
            saved=suite.common.load(folder/'test.pt')
            self.assertEqual(saved['scores'],suite.common.scores(saved['prediction_cycles'],test['label']))
            self.assertEqual(saved['diagnostic_units'],'standardized_natural_log')
            saved['prediction_cycles'][0]+=1;suite.common.save(saved,folder/'test.pt')
            with self.assertRaises(ValueError):suite.completed_rows(root,identity)
            with self.assertRaises(ValueError):suite.evaluate(source,test,p,cfg,folder,binding,mean,scale,'cpu')

    def test_data_preparation_cache_and_raw_routes_counts_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);origin=root/'origin';raw=root/'raw';cache_workspace=root/'cached';raw_workspace=root/'rebuilt'
            (origin/'cache').mkdir(parents=True)
            feature=torch.zeros(6,100,1000)
            groups={chem:[SimpleNamespace(cell_id=f'{chem}_{i:03}',cathode_material=chem,
                nominal_capacity_in_Ah=1.,label=torch.tensor(200.+i),feature=feature)
                for i in range(n)] for chem,n in data.EXPECTED.items()}
            with (origin/'cache/transfer.pkl').open('wb') as f:pickle.dump(groups,f)
            batteries=[b for values in groups.values() for b in values];lookup={}
            for i,b in enumerate(batteries):
                path=raw/'data/processed'/data.SOURCES[i%len(data.SOURCES)]/f'{b.cell_id}.pkl'
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b.cell_id.encode());lookup[path]=b
            def compact_save(battery,material,origin_record,folder):
                path=data.feature_path(folder,battery.cell_id);path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(battery.cell_id.encode())
                return dict(cell_id=battery.cell_id,chemistry=material,label_cycles=float(battery.label),
                    nominal_capacity_in_Ah=1.,feature_file=path.relative_to(folder).as_posix(),
                    feature_sha256=suite.common.digest(path),origin=origin_record)
            with patch.object(data,'save_cell',side_effect=compact_save):
                cached=data.prepare(origin,cache_workspace,{'code':'same'})
                with patch.object(data.BatteryData,'load',side_effect=lambda path:lookup[path]), \
                     patch.object(data.RULLabelAnnotator,'process_cell',side_effect=lambda b:b.label), \
                     patch.object(data.BatLiNetFeatureExtractor,'process_cell',return_value=feature):
                    rebuilt=data.prepare(raw,raw_workspace,{'code':'same'})
            self.assertEqual(cached['observed_counts'],data.EXPECTED)
            self.assertEqual(rebuilt['observed_counts'],data.EXPECTED)
            self.assertEqual([r['cell_id'] for r in cached['records']],[r['cell_id'] for r in rebuilt['records']])
            self.assertFalse(cached['feature_and_label_independently_rebuilt'])
            self.assertTrue(rebuilt['feature_and_label_independently_rebuilt'])
            self.assertEqual(data.prepare(origin,cache_workspace,{'code':'same'}),cached)
            with self.assertRaises(ValueError):data.prepare(origin,cache_workspace,{'code':'changed'})
            (cache_workspace/'dataset'/cached['records'][0]['feature_file']).write_bytes(b'changed')
            with self.assertRaises(ValueError):data.prepare(origin,cache_workspace,{'code':'same'})
            with self.assertRaises(FileNotFoundError):data.prepare(root/'absent',root/'absent_workspace',{})

    def test_saved_feature_shape_label_and_clip_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'dataset'
            feature=torch.zeros(6,100,1000);feature[0,0,0]=11.
            bat=SimpleNamespace(cell_id='test_cell',nominal_capacity_in_Ah=1.,label=torch.tensor(201.),feature=feature)
            r=data.save_cell(bat,'LFP',{},folder)
            part=data.load_part([bat.cell_id],{'records':[r]},root)
            self.assertEqual(float(part['feature'][0,0,0,0]),0.)
            self.assertEqual(float(suite.common.load(folder/r['feature_file'])['feature'][0,0,0]),11.)
            bat.feature=torch.zeros(6,99,1000)
            with self.assertRaises(ValueError):data.save_cell(bat,'LFP',{},folder)
            bat.feature=feature;bat.label=torch.tensor(100.)
            with self.assertRaises(ValueError):data.save_cell(bat,'LFP',{},folder)
            bat.label=torch.tensor(201.);bat.feature=feature.clone();bat.feature[0,0,0]=float('nan')
            with self.assertRaises(ValueError):data.save_cell(bat,'LFP',{},folder)

    def test_full_size_single_pair_backpropagation(self):
        cfg=suite.read_config();model=suite.make_model(cfg,0,'cpu').train()
        rng=torch.Generator().manual_seed(12)
        x=torch.randn(1,6,100,1000,generator=rng)
        ref=torch.randn(1,1,6,100,1000,generator=rng)
        own,sup,_,_,_=model.compute_prediction_components(x,ref,torch.tensor([[.3]]))
        loss=.5*(own-.2).square().mean()+.5*(sup-.2).square().mean();loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))


if __name__=='__main__':
    unittest.main(verbosity=2)
