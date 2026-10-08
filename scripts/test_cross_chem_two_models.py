"""Author-network agreement, two-model protocol and staged restart checks."""
import ast
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
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import original_cross_chem_transfer as original
from scripts import run_cross_chem_two_models as suite
from scripts import run_cross_chem_cycle_soft_difference as combo
from scripts.test_cross_chem_cycle_soft_difference import TransferTests
from scripts import transfer_cycle_soft_data as data


class TwoModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self):
        cfg,source,target,test,p,mean,scale=TransferTests().fixture()
        orig=copy.deepcopy(suite.read_configs()[original.MODEL])
        orig['network'].update(input_height=20,input_width=128,channels=4)
        orig['training'].update(own_epochs=4,relation_epochs=4,relation_patience=10,
            own_batch=4,relation_batch=4,micro_batch=2,inference_chunk=2)
        return orig,cfg,source,target,test,p

    def test_author_cnn_exact_initial_state_forward_and_gradients(self):
        notebook=json.loads((ROOT/'transfer_reproduce/finalized.ipynb').read_text(encoding='utf-8'))
        definitions=[]
        for cell in notebook['cells']:
            if cell['cell_type']!='code':continue
            tree=ast.parse(''.join(cell.get('source',[])))
            definitions += [node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in ('ConvModule','CNNRULPredictor')]
        self.assertEqual(len(definitions),2)
        namespace=dict(torch=torch,nn=nn)
        exec(compile(ast.Module(body=definitions,type_ignores=[]),'author_CNN_classes','exec'),namespace)
        torch.manual_seed(7);a=namespace['CNNRULPredictor'](6,4,20,128)
        torch.manual_seed(7);b=original.CNNRULPredictor(6,4,20,128)
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
        x=torch.randn(3,6,20,128)
        y=a(x);z=b(x);torch.testing.assert_close(y,z,rtol=0,atol=0)
        y.sum().backward();z.sum().backward()
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

    def test_224_model_cases_share_one_protocol_and_label_rules_are_separate(self):
        jobs=suite.plan(['LCO','NCA','NMC'],None);self.assertEqual(len(jobs),224)
        self.assertEqual(len({(c,n,s) for c,n,s,m in jobs}),112)
        for a,b in zip(jobs[::2],jobs[1::2]):
            self.assertEqual(a[:3],b[:3]);self.assertEqual((a[3],b[3]),suite.MODELS)
        self.assertNotIn('label_space',suite.SHARED_POLICY)
        records=TransferTests().records()
        with tempfile.TemporaryDirectory() as tmp:
            p,path=data.protocol({'records':records},'NCA',1,2,Path(tmp),suite.SHARED_POLICY)
            again,path2=data.protocol({'records':records},'NCA',1,2,Path(tmp),suite.SHARED_POLICY)
            self.assertEqual(path,path2);self.assertEqual(p['source_ids'],again['source_ids'])
            data.validate_protocol(p,records,suite.SHARED_POLICY)
        with self.assertRaises(ValueError):suite.plan(['NCA'],[16])

    def test_signed_inputs_clip_after_subtraction_and_do_not_mutate_raw(self):
        cfg,_,source,target,test,p=self.fixture()
        source['feature'].fill_(13.);target['feature'].fill_(15.)
        stats=original.statistics(source,target,p)
        x,_=original.batch(source,target,torch.tensor([0]),'relation',stats,cfg,'cpu')
        self.assertTrue(torch.equal(x,torch.full_like(x,2.)))
        self.assertEqual(float(source['feature'].max()),13.)
        self.assertEqual(float(target['feature'].max()),15.)
        self.assertEqual(float(suite.clip_part(target)['feature'].max()),0.)
        target['feature'].fill_(-4.);target['feature'][:,:,9]=3.
        x=original.own_input(target['feature'],cfg)
        self.assertTrue((x[:,:,9]==0).all());self.assertEqual(float(x[0,0,0,0]),-7.)
        x=original.clean_difference(torch.tensor([-11.,-10.,2.,10.,11.]))
        torch.testing.assert_close(x,torch.tensor([0.,-10.,2.,10.,0.]),rtol=0,atol=0)

    def test_raw_label_statistics_use_train_pairs_and_source_fallback_without_test(self):
        cfg,_,source,target,test,p=self.fixture();stats=original.statistics(source,target,p)
        indices=p['train_pairs'];delta=target['label'][indices//3]-source['label'][indices%3]
        torch.testing.assert_close(stats['delta_mean'],delta.mean(),rtol=0,atol=0)
        torch.testing.assert_close(stats['delta_std'],delta.std(unbiased=True),rtol=0,atol=0)
        torch.testing.assert_close(stats['own_mean'],target['label'].mean(),rtol=0,atol=0)
        self.assertFalse(stats['own_fallback_to_source'])
        small={k:v[:1] for k,v in target.items()};q=dict(p,train_pairs=torch.tensor([0,1]),val_pairs=torch.tensor([2]))
        stats=original.statistics(source,small,q)
        self.assertTrue(stats['own_fallback_to_source'])
        torch.testing.assert_close(stats['own_mean'],source['label'].mean(),rtol=0,atol=0)
        torch.testing.assert_close(stats['own_std'],source['label'].std(unbiased=True),rtol=0,atol=0)
        test['label'].fill_(1e9);second=original.statistics(source,small,q)
        self.assertEqual(float(second['delta_mean']),float(stats['delta_mean']))

    def test_microbatch_gradients_match_whole_batches_including_tail(self):
        cfg,_,source,target,test,p=self.fixture();stats=original.statistics(source,target,p)
        for stage,indices in [('relation',p['train_pairs']),('own',torch.arange(2))]:
            a=original.make_branches(cfg,2,'cpu')[stage];b=original.make_branches(cfg,2,'cpu')[stage]
            oa=torch.optim.SGD(a.parameters(),lr=.001);ob=torch.optim.SGD(b.parameters(),lr=.001)
            full=copy.deepcopy(cfg);full['training']['micro_batch']=4
            x=original.train_epoch(a,source,target,indices,stage,stats,cfg,oa,'cpu')
            y=original.train_epoch(b,source,target,indices,stage,stats,full,ob,'cpu')
            self.assertAlmostEqual(x['train_loss'],y['train_loss'],places=6)
            for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],rtol=1e-5,atol=2e-7)

    def test_relation_early_stop_best_and_own_fixed_final(self):
        cfg,_,source,target,test,p=self.fixture();stats=original.statistics(source,target,p)
        cfg['training'].update(relation_epochs=8,relation_patience=2)
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            folder=Path(tmp)
            with patch.object(original,'validate',side_effect=[1.,.5,.6,.7]):
                relation=original.train_stage(source,target,p,stats,cfg,folder/'relation',{},'relation','cpu')
            self.assertEqual(relation['trained_epochs'],4);self.assertEqual(relation['selected_epoch'],2)
            own=original.train_stage(source,target,p,stats,cfg,folder/'own',{},'own','cpu')
            self.assertEqual(own['selected_epoch'],4);self.assertEqual(own['selection'],'fixed_final_epoch')
            progress=suite.common.load(folder/'relation/progress.pt')
            self.assertTrue(any(not torch.equal(relation['state'][key],progress['state'][key]) for key in relation['state']))
            for key,value in relation['state'].items():torch.testing.assert_close(value,progress['best_state'][key],rtol=0,atol=0)

    def test_original_staged_crash_resume_exact_and_partial_log_removed(self):
        cfg,_,source,target,test,p=self.fixture();stats=original.statistics(source,target,p)
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp)
            for stage in ('relation','own'):
                a=original.train_stage(source,target,p,stats,cfg,root/'a'/stage,{},stage,'cpu')
                save=suite.common.save
                def interrupt(value,path):
                    save(value,path)
                    if Path(path).name=='progress.pt' and value['epoch']==2:raise RuntimeError('simulated crash')
                with patch.object(suite.common,'save',side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError,'simulated crash'):
                        original.train_stage(source,target,p,stats,cfg,root/'b'/stage,{},stage,'cpu')
                with (root/'b'/stage/'train.jsonl').open('a') as f:f.write('{"epoch":3,"train_loss":999}\n')
                b=original.train_stage(source,target,p,stats,cfg,root/'b'/stage,{},stage,'cpu')
                for key,value in a['state'].items():torch.testing.assert_close(value,b['state'][key],rtol=0,atol=0)
                self.assertEqual(a['selected_epoch'],b['selected_epoch'])
                logs=[json.loads(line) for line in (root/'b'/stage/'train.jsonl').read_text().splitlines()]
                self.assertEqual([r['epoch'] for r in logs],[1,2,3,4])
                with self.assertRaises(ValueError):original.train_stage(source,target,p,stats,cfg,root/'b'/stage,{'changed':True},stage,'cpu')

    def test_original_all_reference_predictions_signed_delta_positive_median_raw_fusion(self):
        cfg,_,source,target,test,p=self.fixture();stats=original.statistics(source,target,p)
        branches=original.make_branches(cfg,2,'cpu')
        a=original.predict(branches,source,test,stats,cfg,'cpu')
        cfg2=copy.deepcopy(cfg);cfg2['training']['inference_chunk']=3
        b=original.predict(branches,source,test,stats,cfg2,'cpu')
        self.assertTrue(a['valid'])
        with torch.no_grad():
            features=original.clean_difference(test['feature'][:,None]-source['feature'][None])
            expected=branches['relation'](features.flatten(0,1)).reshape(2,3)*stats['delta_std']+stats['delta_mean']+source['label'][None]
        torch.testing.assert_close(a['diagnostics']['y_sup'],expected,rtol=1e-5,atol=1e-4)
        torch.testing.assert_close(a['prediction_cycles'],b['prediction_cycles'],rtol=1e-5,atol=1e-4)
        values,counts=original.positive_median(torch.tensor([[-10.,0.,10.,20.],[4.,6.,8.,-2.]]))
        torch.testing.assert_close(values,torch.tensor([10.,6.]),rtol=0,atol=0)
        self.assertEqual(counts.tolist(),[2,3])
        with self.assertRaises(FloatingPointError):original.positive_median(torch.tensor([[-1.,0.]]))
        with torch.no_grad():
            branches['relation'].fc.weight.zero_();branches['relation'].fc.bias.fill_(-1e6)
        invalid=original.predict(branches,source,test,stats,cfg,'cpu')
        self.assertFalse(invalid['valid']);self.assertEqual(invalid['failure'],'no_positive_reference_prediction')

    def test_both_models_complete_metrics_receipts_same_ids_and_readonly_report(self):
        cfg,ccfg,source,target,test,p=self.fixture()
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            root=Path(tmp);identity=dict(models=list(suite.MODELS),runs=[['LCO',2,2,m] for m in suite.MODELS])
            suite.common.atomic_json(identity,root/'queue.json')
            suite.common.atomic_json({'fixture':True},root/'dataset/manifest.json')
            ppath=root/'protocols/LCO/train_2/seed_2.pt';suite.common.save(p,ppath)
            rows=[]
            for model in suite.MODELS:
                folder=root/'LCO/train_2'/f'{model}_seed2'
                binding=dict(queue=identity,model=model,target='LCO',train_count=2,seed=2,
                    protocol_sha256=suite.common.digest(ppath),manifest_sha256=suite.common.digest(root/'dataset/manifest.json'))
                if model==original.MODEL:
                    def loader():
                        self.assertTrue((folder/'own/final.pt').exists());self.assertTrue((folder/'relation/final.pt').exists());return test
                    row=original.run(source,target,loader,p,cfg,folder,binding,'cpu')
                    again=original.run(source,target,loader,p,cfg,folder,binding,'cpu');self.assertEqual(row,again)
                else:
                    mean,scale=data.label_statistics(source,target);src=combo.add_z(source,mean,scale);tr=combo.add_z(target,mean,scale)
                    combo.train_run(src,tr,p,ccfg,folder,binding,'cpu');row=combo.evaluate(src,test,p,ccfg,folder,binding,mean,scale,'cpu')
                rows.append(row)
            self.assertEqual(rows,suite.completed_rows(root,identity))
            suite.report(root);self.assertFalse((root/'summary.json').exists())
            a=suite.common.load(root/'LCO/train_2'/f'{original.MODEL}_seed2/test.pt')
            b=suite.common.load(root/'LCO/train_2'/f'{combo.MODEL}_seed2/test.pt')
            self.assertEqual(a['target_test_ids'],b['target_test_ids']);self.assertEqual(a['source_ids'],b['source_ids'])
            torch.testing.assert_close(a['truth_cycles'],b['truth_cycles'],rtol=0,atol=0)
            for output in (a,b):self.assertEqual(output['scores'],suite.common.scores(output['prediction_cycles'],output['truth_cycles']))
            a['prediction_cycles'][0]+=1;suite.common.save(a,root/'LCO/train_2'/f'{original.MODEL}_seed2/test.pt')
            with self.assertRaises(ValueError):suite.completed_rows(root,identity)

    def test_summary_retains_all_seeds_sample_population_std_and_paired_direction(self):
        rows=[]
        for seed in range(8):
            for model in suite.MODELS:
                score=10.+seed if model==original.MODEL else 9.+seed
                rows.append(dict(target='LCO',train_count=1,model=model,seed=seed,
                    selected_epoch=2,trained_epochs=4,RMSE=score,MAE=score,MAPE=score/100,ACC15=1-score/100))
        result=suite.summarize(rows,16);self.assertEqual(len(result['groups']),2)
        m=result['groups'][0]['metrics']['MAPE'];self.assertAlmostEqual(m['sample_std']/m['population_std'],math.sqrt(8/7))
        delta=result['paired_differences'][0]['metrics']
        self.assertAlmostEqual(delta['MAPE']['mean_advantage'],.01);self.assertAlmostEqual(delta['ACC15']['mean_advantage'],.01)
        self.assertEqual(delta['MAPE']['better_seeds'],8)
        self.assertFalse(result['paper_reference']['numerical_values_available'])
        self.assertEqual(len(suite.summarize(rows[:-1],16)['groups']),1)

    def test_both_preflight_paths_use_training_inputs_and_all_references(self):
        cfg,ccfg,source,target,test,p=self.fixture()
        gate=suite.preflight(source,target,p,{original.MODEL:cfg,combo.MODEL:ccfg},'cpu')
        self.assertEqual(gate[original.MODEL]['reference_count'],3)
        self.assertEqual(gate[combo.MODEL]['evaluation_source_count'],3)
        self.assertEqual(gate[original.MODEL]['relation']['optimizer_steps'],2)
        self.assertTrue(math.isfinite(gate[original.MODEL]['own']['train_loss']))

    def test_full_size_original_network_parameters_and_backprop(self):
        cfg=suite.read_configs()[original.MODEL];branches=original.make_branches(cfg,0,'cpu')
        self.assertEqual(sum(p.numel() for m in branches.values() for p in m.parameters()),2274946)
        x=torch.randn(1,6,100,1000);loss=branches['relation'](x).square().mean();loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(branches['relation'].encoder.conv1.weight.grad.abs().sum()>0)


if __name__=='__main__':unittest.main(verbosity=2)
