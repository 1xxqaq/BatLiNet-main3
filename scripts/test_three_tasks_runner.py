"""Synthetic CPU checks; no formal data, training directory or test-set access."""
import argparse
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import run_three_tasks as r


def small_config(name):
    c = copy.deepcopy(r.configs('matr_1')[name])
    c['model'].update(input_height=20, input_width=128, channels=4, epochs=2, train_batch_size=4)
    if name != 'batlinet':
        c['model'].update(attention_channels=8, attention_heads=2, head_hidden_channels=8)
    c['runner'].update(micro_batch_size=2, checkpoint_every=1, pair_chunk=8)
    return c


def fixture():
    g = torch.Generator().manual_seed(123)
    data = {}
    for part, n in (('train',5),('test',3)):
        data[part] = dict(feature=torch.randn(n,6,20,128,generator=g),
                          label=torch.linspace(150,350,n), ids=[f'{part}_{i}' for i in range(n)])
    logs=data['train']['label'].log()
    data.update(mean=logs.mean(),scale=logs.std())
    return data


def binding(cfg,name):
    return dict(config=cfg,model=name,seed=3,fixture=True)


def check_configs():
    counts={}
    for task in r.TASKS:
        cfgs=r.configs(task)
        for name,cfg in cfgs.items():
            m=cfg['model']
            r.FEATURE_EXTRACTORS.build(cfg['feature'])
            r.LABEL_ANNOTATORS.build(cfg['label'])
            assert m['train_support_size']==(1 if task=='matr_2' else 2)
            assert m['alpha']==(.2 if task=='matr_2' else .5)
            assert (m['input_height'],m['input_width'])==(100,1000)
            assert m['filter_cycles'] is False and m['epochs']==1000
            assert cfg['runner']['micro_batch_size']==16
            r.seed_all(0)
            model=r.MODELS.build(m,seed=0)
            counts[task,name]=sum(p.numel() for p in model.parameters())
            if name=='latent_cycle_mixer':
                assert model.cell_encoder.cycle_mixer.kind=='mlp'
            if name=='batlinet':
                assert model.diff_base==(10 if task=='hust' else 9)
            else:
                assert 'diff_base' not in m
    for task in r.TASKS:
        assert counts[task,'latent_cross_attention']==209890
        assert counts[task,'latent_cycle_mixer']==211556
    # Zero-initialized residual and isolated auxiliary RNG initialization.
    r.seed_all(7)
    base=r.MODELS.build(r.configs('matr_1')['latent_cross_attention']['model'],seed=7)
    rng=torch.get_rng_state().clone()
    r.seed_all(7)
    enhanced=r.MODELS.build(r.configs('matr_1')['latent_cycle_mixer']['model'],seed=7)
    assert torch.equal(rng,torch.get_rng_state())
    for key,value in base.state_dict().items():
        target=key.replace('cell_encoder.','cell_encoder.base.',1) if key.startswith('cell_encoder.') else key
        assert torch.equal(value,enhanced.state_dict()[target]),key
    return {f'{t}/{n}':v for (t,n),v in counts.items()}


def check_prediction(model,data,indices,temp):
    chunked=r.predict(model,data,indices,'cpu',8)
    whole=r.predict(model,data,indices,'cpu',32)
    torch.testing.assert_close(chunked['prediction'],whole['prediction'],atol=1e-4,rtol=1e-5)
    torch.testing.assert_close(chunked['diagnostics']['y_sup'],whole['diagnostics']['y_sup'],atol=1e-5,rtol=1e-5)
    # Compare with the EXISTING model's public prediction entry, not a second new implementation.
    ds_train=r.Dataset(data['train']['feature'],(data['train']['label'].log()-data['mean'])/data['scale'])
    ds_test=r.Dataset(data['test']['feature'],(data['test']['label'].log()-data['mean'])/data['scale'])
    model.fixed_test_support_index_path=str(temp/'references.pt')
    model._fixed_test_support_index=indices
    prediction,diagnostics=model.predict(SimpleNamespace(train_data=ds_train,test_data=ds_test),return_diagnostics=True)
    expected=(prediction.cpu()*data['scale']+data['mean']).exp()
    torch.testing.assert_close(chunked['prediction'],expected,atol=1e-4,rtol=1e-5)
    assert torch.equal(diagnostics['support_index'],indices)


def check_weighting():
    # A deterministic scalar model verifies short final microbatch/logical batch weighting.
    class Scalar(torch.nn.Module):
        train_support_size=2
        def __init__(self):
            super().__init__(); self.p=torch.nn.Parameter(torch.tensor(.7))
        def get_support_set(self,q,pool,labels,**kw): return q,labels
        def forward(self,x,y,*_,**kw): return ((self.p*x.flatten()-y)**2).mean()
    tensors=(torch.arange(1.,8.),)*3+(torch.arange(2.,9.),)
    a,b=Scalar(),Scalar()
    oa,ob=torch.optim.SGD(a.parameters(),lr=.001),torch.optim.SGD(b.parameters(),lr=.001)
    r.train_epoch(a,oa,tensors,torch.Generator().manual_seed(0),5,2)
    r.train_epoch(b,ob,tensors,torch.Generator().manual_seed(0),5,5)
    torch.testing.assert_close(a.p,b.p,rtol=1e-6,atol=1e-7)


def check_run(name,temp,data):
    cfg=small_config(name)
    bind=binding(cfg,name)
    full=temp/name/'full'
    interrupted=temp/name/'interrupted'
    r.train_run(data,cfg,3,full,bind,'cpu')
    # Inject a crash AFTER a durable epoch-one checkpoint, before epoch two completes.
    original_save=r.save
    crashed=False
    def crash_save(value,path):
        nonlocal crashed
        original_save(value,path)
        if Path(path).name=='progress.pt' and value.get('epoch')==1 and not crashed:
            crashed=True
            raise RuntimeError('synthetic interruption')
    r.save=crash_save
    try:
        try:
            r.train_run(data,cfg,3,interrupted,bind,'cpu')
        except RuntimeError as e:
            assert str(e)=='synthetic interruption'
    finally:
        r.save=original_save
    assert crashed
    # Simulate an additional unsaved/torn epoch log from the interrupted interval.
    with (interrupted/'train.jsonl').open('a') as f:
        f.write('{"epoch":2,"train_loss":0.123}\n{"epoch":')
    r.train_run(data,cfg,3,interrupted,bind,'cpu')
    expected=r.load(full/'final.pt')
    actual=r.load(interrupted/'final.pt')
    for k,v in expected['state'].items(): assert torch.equal(v,actual['state'][k]),k
    assert list(interrupted.glob('interrupted_log_*.jsonl'))
    cp1=r.load(full/'progress.pt');cp2=r.load(interrupted/'progress.pt')
    assert torch.equal(cp1['rng']['references'],cp2['rng']['references'])
    assert torch.equal(cp1['rng']['torch'],cp2['rng']['torch'])
    before=r.digest(interrupted/'final.pt')
    r.train_run(data,cfg,3,interrupted,bind,'cpu')
    assert before==r.digest(interrupted/'final.pt')
    try:
        r.train_run(data,cfg,3,interrupted,dict(bind,seed=4),'cpu')
    except ValueError: pass
    else: raise AssertionError('configuration mismatch was accepted')
    model=r.MODELS.build(cfg['model'],seed=3)
    model.load_state_dict(actual['state'],strict=True)
    indices=r.protocol(data,3,temp/name/'references.pt')
    check_prediction(model,data,indices,temp/name)
    first=r.evaluate_run(data,cfg,3,interrupted,bind,indices,'cpu')
    assert first==r.evaluate_run(data,cfg,3,interrupted,bind,indices,'cpu')
    indices[0,0]=(indices[0,0]+1)%len(data['train']['ids'])
    try: r.evaluate_run(data,cfg,3,interrupted,bind,indices,'cpu')
    except ValueError: pass
    else: raise AssertionError('different reference indices were accepted')
    stored=r.load(temp/name/'references.pt')
    stored['indices'][0,0]=(stored['indices'][0,0]+1)%len(data['train']['ids'])
    r.save(stored,temp/name/'references.pt')
    try: r.protocol(data,3,temp/name/'references.pt')
    except ValueError: pass
    else: raise AssertionError('modified fixed protocol was accepted')
    return dict(resume_bitwise_equal=True, completed_skip=True, prediction_matches_existing_entry=True,
                chunked_median_matches_all_32=True, wrong_binding_and_references_rejected=True)


def main():
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    counts=check_configs()
    check_weighting()
    data=fixture()
    r.validate_data(data,(6,20,128))
    with tempfile.TemporaryDirectory(prefix='batlinet_three_tasks_') as tmp:
        results={name:check_run(name,Path(tmp),data) for name in r.MODELS_NAMES}
    report=dict(status='passed',synthetic_only=True,formal_data_checked=False,
                cuda_checked=False,runtime=r.runtime('cpu'),counts=counts,checks=results,
                tail_batch_sample_weighting=True,source_code=r.source_hashes())
    if args.report:
        r.atomic_json(report,args.report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
