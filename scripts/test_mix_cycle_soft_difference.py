"""Combination structure, legacy-loop agreement and crash-recovery CPU tests."""
import copy
import io
import json
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
from scripts import run_mix_cycle_soft_difference as suite
from src.data.databundle import Dataset, DataBundle
from src.data.transformation.sequential import SequentialDataTransformation
from src.models.rul_predictors.soft_aligned_difference_batlinet import SoftAlignedDifferenceBlock


class CombinationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def fixture(self):
        generator=torch.Generator().manual_seed(23)
        d={}
        mean,scale=torch.tensor(5.3),torch.tensor(.5)
        for part,n in (('train',5),('test',3)):
            z=torch.linspace(-.2,.8,n)
            d[part]=dict(feature=torch.randn(n,6,20,128,generator=generator),label_z=z,
                         label=(z*scale+mean).exp(),ids=[f'{part}_{i}' for i in range(n)])
        return dict(d,mean=mean,scale=scale)

    def small(self):
        c=copy.deepcopy(suite.model_config('mix_20'))
        c['model'].update(input_width=128,train_batch_size=4,epochs=2,attention_channels=8,
                          attention_heads=2,head_hidden_channels=8,evaluate_freq=1)
        c['runner'].update(micro_batch_size=2,checkpoint_every=1,periodic_rng_every=1,pair_chunk=8)
        return c

    def test_initial_zero_equivalence_parameter_counts_rng_and_tokens(self):
        for task,expected,tokens in (('mix_20',170556,155),('mix_100',211556,775)):
            c=suite.model_config(task)['model']
            soft={k:v for k,v in c.items() if k!='cycle_mixer_hidden'}
            soft['name']='SoftAlignedDifferenceBatLiNetRULPredictor'
            torch.manual_seed(8);a=suite.common.MODELS.build(soft,seed=8).eval();rng=torch.get_rng_state().clone()
            torch.manual_seed(8);b=suite.common.MODELS.build(c,seed=8).eval()
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(sum(p.numel() for p in b.parameters()),expected)
            self.assertEqual(b.cell_encoder.num_tokens,tokens)
            self.assertIsInstance(b.cross_attention[0],SoftAlignedDifferenceBlock)
            self.assertEqual(len(b.cross_attention),1)
            for k,v in a.state_dict().items():
                target=k.replace('cell_encoder.','cell_encoder.base.',1) if k.startswith('cell_encoder.') else k
                self.assertTrue(torch.equal(v,b.state_dict()[target]),k)
            x=torch.randn(1,6,c['input_height'],1000)
            y=torch.randn(1,1)
            with torch.no_grad():
                aa=a.compute_prediction_components(x,x[:,None],y)
                bb=b.compute_prediction_components(x,x[:,None],y)
            for v,w in zip(aa[:3],bb[:3]):torch.testing.assert_close(v,w,rtol=0,atol=0)

    def test_residual_output_and_difference_projections_learn(self):
        c=self.small();model=suite.common.MODELS.build(c['model'],seed=5)
        x=torch.randn(2,6,20,128,requires_grad=True)
        ref=torch.randn(2,2,6,20,128,requires_grad=True)
        labels=torch.randn(2);rlabels=torch.randn(2,2)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        loss=model(x,labels,ref,rlabels,return_loss=True);loss.backward()
        residual=model.cell_encoder.cycle_mixer.net
        self.assertGreater(float(residual[-1].weight.grad.abs().sum()),0)
        self.assertEqual(float(residual[0].weight.grad.abs().sum()),0)
        for grad in (x.grad,ref.grad,*model.cross_attention[0].attn.in_proj_weight.grad.chunk(3)):
            self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.abs().sum()),0)
        optimizer.step();optimizer.zero_grad()
        model(x,labels,ref,rlabels,return_loss=True).backward()
        self.assertGreater(float(residual[0].weight.grad.abs().sum()),0)

    def test_epoch_matches_existing_full_and_micro_loops_including_monitor_rng(self):
        data=self.fixture();cfg=self.small()
        for micro in (128,2):
            suite.historical20.set_seed(11)
            old=suite.common.MODELS.build(cfg['model'],seed=11)
            training=Dataset(data['train']['feature'].clone(),data['train']['label_z'].clone())
            if micro==128:
                suite.historical20.legacy_train(old,training,1,1,len(data['test']['ids']))
            else:
                suite.historical100.train(old,training,1,micro,len(data['test']['ids']))
            expected_rng=torch.get_rng_state().clone()
            suite.historical20.set_seed(11)
            new=suite.common.MODELS.build(cfg['model'],seed=11)
            ds=suite.prepared(new,data['train'],'cpu')
            optimizer=torch.optim.AdamW(new.parameters(),lr=.001,weight_decay=.01)
            row=suite.train_epoch(new,optimizer,ds,micro,1,1,len(data['test']['ids']))
            self.assertEqual(row['optimizer_steps'],2)
            self.assertTrue(torch.equal(expected_rng,torch.get_rng_state()))
            for key,val in old.state_dict().items():
                torch.testing.assert_close(val,new.state_dict()[key],atol=0,rtol=0,msg=key)

    def test_resume_skip_and_result_corruption(self):
        data=self.fixture();cfg=self.small();binding=dict(config=cfg,task='synthetic',seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            suite.train_run(data,cfg,3,a,binding,'cpu')
            original=suite.common.save
            crashed=False
            def crash(value,path):
                nonlocal crashed
                original(value,path)
                if Path(path).name=='progress.pt' and value.get('epoch')==1 and not crashed:
                    crashed=True;raise RuntimeError('synthetic crash')
            with patch.object(suite.common,'save',side_effect=crash):
                with self.assertRaisesRegex(RuntimeError,'synthetic crash'):
                    suite.train_run(data,cfg,3,b,binding,'cpu')
            with (b/'train.jsonl').open('a') as log:log.write('{"epoch":2,"train_loss":0.1}\n{"epoch":')
            suite.train_run(data,cfg,3,b,binding,'cpu')
            aa,bb=suite.common.load(a/'final.pt'),suite.common.load(b/'final.pt')
            for key,val in aa['state'].items():self.assertTrue(torch.equal(val,bb['state'][key]),key)
            pa,pb=suite.common.load(a/'progress.pt'),suite.common.load(b/'progress.pt')
            self.assertTrue(torch.equal(pa['rng']['torch'],pb['rng']['torch']))
            self.assertTrue(list(b.glob('interrupted_log_*.jsonl')))
            sha=suite.common.digest(b/'final.pt')
            suite.train_run(data,cfg,3,b,binding,'cpu')
            self.assertEqual(sha,suite.common.digest(b/'final.pt'))
            with self.assertRaises(ValueError):suite.train_run(data,cfg,3,b,dict(binding,seed=4),'cpu')
            indices=torch.arange(96).reshape(3,32)%5
            one=suite.evaluate_run(data,cfg,3,b,binding,indices,'cpu')
            self.assertEqual(one,suite.evaluate_run(data,cfg,3,b,binding,indices,'cpu'))
            result=suite.common.load(b/'test.pt');result['diagnostics']['support_index'][0,0]=4
            suite.common.save(result,b/'test.pt')
            with self.assertRaises(ValueError):suite.evaluate_run(data,cfg,3,b,binding,indices,'cpu')

    def test_chunked_prediction_agrees_existing_entry_and_full_32(self):
        data=self.fixture();cfg=self.small()
        model=suite.common.MODELS.build(cfg['model'],seed=7)
        indices=torch.arange(96).reshape(3,32)%5
        split=suite.predict(model,data,indices,'cpu',8)
        full=suite.predict(model,data,indices,'cpu',32)
        torch.testing.assert_close(split['prediction'],full['prediction'],rtol=1e-5,atol=1e-6)
        bundle=SimpleNamespace(train_data=Dataset(data['train']['feature'],data['train']['label_z']),
                               test_data=Dataset(data['test']['feature'],data['test']['label_z']))
        model.fixed_test_support_index_path='unused';model._fixed_test_support_index=indices
        expected,diag=model.predict(bundle,return_diagnostics=True)
        torch.testing.assert_close(split['prediction'],expected,rtol=1e-5,atol=1e-6)
        self.assertTrue(torch.equal(diag['support_index'],indices))

    def test_frozen_history_cache_and_reference_integrity(self):
        transform=SequentialDataTransformation([dict(name='LogScaleDataTransformation'),dict(name='ZScoreDataTransformation')])
        bundle=DataBundle(torch.randn(3,6,20,1000),torch.tensor([150.,250.,350.]),
                          torch.randn(2,6,20,1000),torch.tensor([200.,300.]),
                          train_metadata=[dict(cell_id=f'a{i}') for i in range(3)],
                          test_metadata=[dict(cell_id=f'b{i}') for i in range(2)],label_transformation=transform)
        bundle.train_data.feature[:,:,10]=0
        bundle.test_data.feature[:,:,10]=0
        protocols=[torch.arange(64).reshape(2,32)%3 for _ in range(8)]
        with tempfile.TemporaryDirectory() as tmp,patch.object(suite.historical20,'audit_history',return_value=(bundle,protocols,[],dict(fixture=True))):
            workspace=Path(tmp)
            d,p,sha=suite.prepare_history('mix_20',Path('unused'),Path('unused'),workspace)
            again,_,sha2=suite.prepare_history('mix_20',Path('unused'),Path('unused'),workspace)
            self.assertEqual(sha,sha2)
            self.assertTrue(torch.equal(d['train']['label_z'],bundle.train_data.label))
            fixed=workspace/'protocols/mix_20/seed_0.pt'
            corrupted=suite.common.load(fixed);corrupted['indices'][0,0]=2;suite.common.save(corrupted,fixed)
            with self.assertRaises(ValueError):suite.prepare_history('mix_20',Path('unused'),Path('unused'),workspace)


    def test_optional_old_reference_is_verified_and_rejected_when_changed(self):
        data=self.fixture();data['historical_binding']=dict(dataset=dict(fixture=True))
        cfg=self.small()['model'];cfg.pop('cycle_mixer_hidden')
        cfg['name']='SoftAlignedDifferenceBatLiNetRULPredictor'
        model=suite.common.MODELS.build(cfg,seed=0)
        indices=torch.arange(96).reshape(3,32)%5
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);workspace=root/'new'
            for seed in range(8):
                folder=root/'workspaces/mix20_soft_aligned_difference_207_v1'/f'latent_soft_aligned_difference_seed{seed}'
                name='latent_soft_aligned_difference'
                suite.common.save(dict(epoch=1000,seed=seed,model=name,config=cfg,state=model.state_dict(),provenance=data['historical_binding']),folder/'epoch1000.pt')
                z=torch.zeros(3);cycles=(z*data['scale']+data['mean']).exp()
                result=dict(epoch=1000,seed=seed,model=name,provenance=data['historical_binding'],
                    checkpoint_sha256=suite.common.digest(folder/'epoch1000.pt'),prediction=z,prediction_cycles=cycles,
                    truth_cycles=data['test']['label'],diagnostics=dict(support_index=indices),scores=suite.common.scores(cycles,data['test']['label']))
                suite.common.save(result,folder/'test.pt')
                suite.common.save(dict(indices=indices),workspace/'protocols/mix_20'/f'seed_{seed}.pt')
            with patch.object(suite,'ROOT',root):
                rows=suite.existing_references('mix_20',data,workspace)
                self.assertEqual(len(rows),8)
                path=root/'workspaces/mix20_soft_aligned_difference_207_v1/latent_soft_aligned_difference_seed0/test.pt'
                r=suite.common.load(path);r['diagnostics']['support_index'][0,0]=4;suite.common.save(r,path)
                rows=suite.existing_references('mix_20',data,workspace)
                self.assertEqual(rows,[])



if __name__=='__main__':
    test_suite=unittest.defaultTestLoader.loadTestsFromTestCase(CombinationTests)
    result=unittest.TextTestRunner(verbosity=2).run(test_suite)
    if '--report' in sys.argv:
        dest=Path(sys.argv[sys.argv.index('--report')+1])
        suite.common.atomic_json(dict(status='passed' if result.wasSuccessful() else 'failed',
            tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
            synthetic_only=True,formal_training_started=False,cuda_checked=False,
            runtime=suite.runtime('cpu'),source_code=suite.sources()),dest)
    sys.exit(0 if result.wasSuccessful() else 1)
