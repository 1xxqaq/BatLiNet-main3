"""Run only the cycle + soft-difference combination on two frozen MIX tasks."""
import argparse
import copy
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_three_tasks as common
from scripts import run_mix20_cycle_mixer_207 as historical20
from scripts import run_mix100_cycle_mixer as historical100
# Explicit registration leaves historical entry points/package imports unchanged.
from src.models.rul_predictors import cycle_mixer_soft_aligned_difference_batlinet
from src.data.databundle import Dataset

TASKS = ('mix_20', 'mix_100')
MODEL = 'latent_cycle_soft_difference'


def model_config(task):
    cfg = yaml.safe_load((ROOT / f'configs/mix_cycle_soft_difference_v1/{task}/{MODEL}.yaml').read_text(encoding='utf-8'))
    m, r = cfg['model'], cfg['runner']
    expected_runner=dict(schema=1,seeds=list(range(8)),micro_batch_size=128 if task=='mix_20' else 16,
        pair_chunk=32 if task=='mix_20' else 8,precision='float32',shuffle=False,optimizer='AdamW',
        weight_decay=.01,checkpoint_every=25,selection='fixed_epoch_1000',label_std='historical_sample',
        train_reference_rng='global_torch_rng_on_training_device',test_reference='reuse_historical_actual_indices',
        test_during_training=False,compile=False,periodic_rng_compensation=True,periodic_rng_every=100)
    if r != expected_runner:
        raise ValueError('配置中的训练策略与实际执行入口不同，需另建协议。')
    if (m['name'] != 'CycleMixerSoftAlignedDifferenceBatLiNetRULPredictor'
            or m['epochs'] != 1000 or m['input_height'] != (20 if task=='mix_20' else 100)
            or m['input_width'] != 1000 or m['train_batch_size'] != 128
            or m['train_support_size'] != 2 or m['test_support_size'] != 32
            or m['alpha'] != .5 or m['filter_cycles'] is not False
            or m['gradient_accumulation_steps'] != 1 or m['cycle_mixer_hidden'] != 16
            or r['micro_batch_size'] != (128 if task=='mix_20' else 16)
            or r['pair_chunk'] != (32 if task=='mix_20' else 8)
            or r['seeds'] != list(range(8)) or r['periodic_rng_every'] != 100
            or r['periodic_rng_compensation'] is not True):
        raise ValueError('配置偏离本次仅组合版、两任务八种子的冻结协议。')
    return cfg


def sources():
    names = ['scripts/run_mix_cycle_soft_difference.py', 'scripts/run_three_tasks.py',
             'scripts/run_mix20_cycle_mixer_207.py', 'scripts/run_mix100_cycle_mixer.py', 'scripts/pipeline.py']
    paths = [ROOT / n for n in names] + list((ROOT / 'src').rglob('*.py'))
    paths += list((ROOT / 'configs/mix_cycle_soft_difference_v1').rglob('*.yaml'))
    return {p.relative_to(ROOT).as_posix(): common.digest(p) for p in sorted(paths)}


def runtime(device):
    value = common.runtime(device)
    value['tf32'] = dict(cudnn=torch.backends.cudnn.allow_tf32, matmul=torch.backends.cuda.matmul.allow_tf32)
    return value


def prepare_history(task, history_root, original_root, workspace):
    if task == 'mix_20':
        bundle, protocols, rows, history = historical20.audit_history(history_root, original_root)
    else:
        bundle, protocols, rows, history = historical100.audit_history(history_root)
    transform = bundle.label_transformation
    standardizer = transform.transformations[1]
    data = dict(mean=standardizer._mean.detach().cpu().reshape(()),
                scale=standardizer._std.detach().cpu().reshape(()), historical_binding=history)
    for part in ('train','test'):
        ds = getattr(bundle, part+'_data')
        data[part] = dict(feature=ds.feature.detach().cpu(), label_z=ds.label.detach().cpu().reshape(-1),
                          label=transform.inverse_transform(ds.label).detach().cpu().reshape(-1),
                          ids=[record['cell_id'] for record in ds.metadata])
    common.validate_data(data, (6, 20 if task=='mix_20' else 100, 1000))
    if task=='mix_20' and any(torch.count_nonzero(data[part]['feature'][:,:,10]).item() for part in ('train','test')):
        raise ValueError('MIX-20历史输入第11循环未置零，与已声明输入协议不符。')
    for part in ('train','test'):
        torch.testing.assert_close(data[part]['label_z'], (data[part]['label'].log()-data['mean'])/data['scale'])
    folder = workspace / 'datasets' / task
    path = folder / 'data.pt'
    receipt_path = folder / 'receipt.json'
    if path.exists():
        stored = common.load(path)
        if stored['historical_binding'] != history:
            raise ValueError(f'已有缓存的历史来源不同：{task}')
        for part in ('train','test'):
            if stored[part]['ids'] != data[part]['ids']:
                raise ValueError(f'缓存电池编号不同：{task}')
            for key in ('feature','label','label_z'):
                if not torch.equal(stored[part][key], data[part][key]):
                    raise ValueError(f'缓存内容被修改：{task}/{part}/{key}')
        for key in ('mean','scale'):
            if not torch.equal(stored[key],data[key]):
                raise ValueError('缓存标签统计量被修改。')
        del stored
    else:
        common.save(data,path)
    sha = common.digest(path)
    if receipt_path.exists() and json.loads(receipt_path.read_text(encoding='utf-8'))['sha256'] != sha:
        raise ValueError('数据文件指纹被修改。')
    common.atomic_json(dict(sha256=sha, historical_binding=history, train_ids=data['train']['ids'],
                           test_ids=data['test']['ids'], train_count=len(data['train']['ids']),
                           test_count=len(data['test']['ids']), label_mean=float(data['mean']),
                           label_sample_std=float(data['scale'])), receipt_path)
    for seed, indices in enumerate(protocols):
        fixed = dict(seed=seed, train_ids=data['train']['ids'], test_ids=data['test']['ids'], indices=indices.cpu())
        dest = workspace / 'protocols' / task / f'seed_{seed}.pt'
        if dest.exists():
            actual = common.load(dest)
            if any(actual[k] != fixed[k] for k in ('seed','train_ids','test_ids')) or not torch.equal(actual['indices'],indices):
                raise ValueError(f'已有参考不等于历史实际参考：{dest}')
        else:
            common.save(fixed,dest)
    # Historical observations are saved separately, never merged into new-run averages.
    common.atomic_json(rows, folder / 'historical_per_seed.json')
    return data, protocols, sha



def existing_references(task,data,workspace):
    """Read existing results only. Never train or promote historical observations."""
    folders = [('mix20_soft_aligned_difference_207_v1','latent_original_matched'),
               ('mix20_soft_aligned_difference_207_v1','latent_soft_aligned_difference'),
               ('mix20_cycle_mixer_207_v1','latent_cycle_mixer')] if task=='mix_20' else [
               ('mix100_cycle_mixer_v1','latent_cross_attention_current'),
               ('mix100_cycle_mixer_v1','latent_cycle_mixer')]
    rows,receipts=[],[]
    for dirname,name in folders:
        records=[]
        try:
            for seed in range(8):
                folder=ROOT/'workspaces'/dirname/f'{name}_seed{seed}'
                path=folder/'test.pt';weight=folder/'epoch1000.pt'
                if not path.is_file() or not weight.is_file():
                    raise FileNotFoundError(f'缺少历史预测或权重：{folder}')
                result=historical20.load(path)
                provenance=result['provenance']
                if provenance['dataset'] != data['historical_binding']['dataset']:
                    raise ValueError('历史电池、顺序、输入或标签变换不一致')
                if result['model']!=name or result['seed']!=seed or result['epoch']!=1000:
                    raise ValueError('历史模型、种子或轮次不同')
                if result['checkpoint_sha256']!=common.digest(weight):
                    raise ValueError('历史权重指纹不同')
                cp=historical20.load(weight)
                if cp['epoch']!=1000 or cp['seed']!=seed or cp['model']!=name or cp['provenance']!=provenance:
                    raise ValueError('历史预测与权重元数据绑定不同')
                expected_class={'latent_original_matched':'LatentCrossAttentionBatLiNetRULPredictor',
                    'latent_soft_aligned_difference':'SoftAlignedDifferenceBatLiNetRULPredictor',
                    'latent_cross_attention_current':'LatentCrossAttentionBatLiNetRULPredictor',
                    'latent_cycle_mixer':'CycleMixerLatentCrossAttentionBatLiNetRULPredictor'}[name]
                if cp['config']['name'] != expected_class:
                    raise ValueError('历史模型名称与实际结构不符')
                model=common.MODELS.build(dict(cp['config']),seed=seed)
                model.load_state_dict(cp['state'],strict=True)
                fixed=common.load(workspace/'protocols'/task/f'seed_{seed}.pt')['indices']
                if not torch.equal(result['diagnostics']['support_index'],fixed):
                    raise ValueError('历史实际参考不同')
                if not torch.equal(result['truth_cycles'].reshape(-1),data['test']['label']):
                    raise ValueError('历史真值不同')
                torch.testing.assert_close(result['prediction_cycles'],(result['prediction']*data['scale']+data['mean']).exp())
                measured=common.scores(result['prediction_cycles'].reshape(-1),data['test']['label'])
                if any(not math.isclose(measured[k],result['scores'][k],rel_tol=1e-5,abs_tol=1e-5) for k in measured):
                    raise ValueError('历史指标不能复算')
                records.append(dict(task=task,model=name,seed=seed,**measured))
                receipts.append(dict(task=task,model=name,seed=seed,status='verified_reference_only',
                                     test_sha256=common.digest(path),checkpoint_sha256=result['checkpoint_sha256'],
                                     runtime_recorded='runtime' in provenance))
            rows.extend(records)
        except Exception as error:
            # Optional reference corruption must not silently become a valid comparison.
            receipts.append(dict(task=task,model=name,status='unavailable_or_rejected',reason=str(error)))
            print(f'已有参考未纳入：{task}/{name}：{error}',flush=True)
    common.atomic_json(rows,workspace/'datasets'/task/'additional_historical_per_seed.json')
    common.atomic_json(receipts,workspace/'datasets'/task/'historical_reference_receipts.json')
    return rows


def prepared(model, part, device):
    ds = Dataset(part['feature'].to(device), part['label_z'].to(device))
    return model.build_cell_dataset(ds)


def train_epoch(model, optimizer, ds, micro, monitor_every, epoch, num_test):
    model.train()
    loader = DataLoader(ds, model.train_batch_size, shuffle=False)
    total, steps, micros = 0., 0, 0
    for batch in loader:
        x, y = batch['feature'], batch['label']
        indices = torch.randint(len(ds), (len(x)*model.train_support_size,), device=ds.device).view(len(x),-1)
        optimizer.zero_grad(set_to_none=True)
        for start in range(0,len(x),micro):
            end = min(start+micro,len(x))
            sx,sy = model.get_support_set(x[start:end],ds.feature,ds.label,
                                          fixed_indices=indices[start:end],support_is_prepared=True)
            loss = model(x[start:end],y[start:end],sx,sy,return_loss=True)
            if not torch.isfinite(loss):
                raise FloatingPointError('训练损失非有限。')
            (loss*((end-start)/len(x))).backward()
            total += loss.detach().item()*(end-start)
            micros += 1
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('梯度非有限。')
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('优化器更新后参数非有限。')
        steps += 1
    if epoch % monitor_every == 0:
        model.eval()
        historical20.advance_old_monitor_rng(len(ds),num_test,model.test_support_size,model.test_batch_size,ds.device)
    return dict(train_loss=total/len(ds),optimizer_steps=steps,micro_batches=micros)


def capture_rng(device):
    value = common.rng_state(torch.Generator().manual_seed(0),device)
    del value['references']  # Training references use the saved global device RNG.
    return value


def restore_rng(value, device):
    dummy = torch.Generator().manual_seed(0)
    common.restore_rng(dict(value,references=dummy.get_state()),dummy,device)


def train_run(data,cfg,seed,folder,binding,device):
    folder.mkdir(parents=True,exist_ok=True)
    historical20.set_seed(seed)
    model = common.MODELS.build(cfg['model'],seed=seed).to(device)
    ds = prepared(model,data['train'],device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['model']['lr'],weight_decay=cfg['runner']['weight_decay'])
    final, progress = folder/'final.pt',folder/'progress.pt'
    start = 0
    if final.exists():
        saved = common.checked_checkpoint(final,binding)
        if saved['epoch'] != cfg['model']['epochs']:
            raise ValueError('最终权重轮次错误。')
        common.align_log(folder,saved['epoch'])
        model.load_state_dict(saved['state'],strict=True)
        info_path = folder/'run.json'
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding='utf-8'))
            if info.get('final_sha256',common.digest(final)) != common.digest(final):
                raise ValueError('已完成权重指纹被修改。')
        print(f'跳过已核验训练：{folder.parent.name}/{folder.name}',flush=True)
        return
    if progress.exists():
        saved = common.checked_checkpoint(progress,binding)
        start = saved['epoch']
        model.load_state_dict(saved['state'],strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'],device)
    else:
        if (folder/'run.json').exists() and json.loads((folder/'run.json').read_text(encoding='utf-8'))['binding'] != binding:
            raise ValueError('未完成目录属于不同设置。')
        common.save(dict(epoch=0,binding=binding,state=model.state_dict(),optimizer=optimizer.state_dict(),rng=capture_rng(device)),progress)
    common.align_log(folder,start)
    common.atomic_json(dict(status='training',binding=binding,parameters=sum(p.numel() for p in model.parameters())),folder/'run.json')
    with (folder/'train.jsonl').open('a',encoding='utf-8') as log:
        for epoch in range(start+1,cfg['model']['epochs']+1):
            tick=time.monotonic()
            row=train_epoch(model,optimizer,ds,cfg['runner']['micro_batch_size'],cfg['runner']['periodic_rng_every'],epoch,len(data['test']['ids']))
            row.update(epoch=epoch,seconds=time.monotonic()-tick,
                       cuda_peak_GiB=torch.cuda.max_memory_allocated(device)/2**30 if device.startswith('cuda') else 0.)
            log.write(json.dumps(row)+'\n');log.flush()
            if epoch % cfg['runner']['checkpoint_every']==0 or epoch==cfg['model']['epochs']:
                os.fsync(log.fileno())
                common.save(dict(epoch=epoch,binding=binding,state=model.state_dict(),optimizer=optimizer.state_dict(),rng=capture_rng(device)),progress)
                common.atomic_json(dict(status='training',task=binding['task'],model=MODEL,seed=seed,epoch=epoch),folder.parent.parent/'status.json')
                print(f'{binding["task"]} 组合版种子{seed} [{epoch}/{cfg["model"]["epochs"]}] 损失 {row["train_loss"]:.6f}，本轮 {row["seconds"]:.1f}秒',flush=True)
    common.save(dict(epoch=cfg['model']['epochs'],binding=binding,state=model.state_dict()),final)
    common.atomic_json(dict(status='trained',binding=binding,final_sha256=common.digest(final)),folder/'run.json')


@torch.no_grad()
def predict(model,data,indices,device,chunk):
    model.eval()
    training=prepared(model,data['train'],device)
    query=prepared(model,data['test'],device)
    origins,refs=[],[]
    if indices.shape != (len(query),32) or indices.min()<0 or indices.max()>=len(training):
        raise ValueError('固定32参考形状或范围错误。')
    for i in range(len(query)):
        parts=[]
        for start in range(0,32,chunk):
            sx,sy=model.get_support_set(query.feature[i:i+1],training.feature,training.label,
                                       fixed_indices=indices[i:i+1,start:start+chunk],support_is_prepared=True)
            ori,sup,*_=model.compute_prediction_components(query.feature[i:i+1],sx,sy)
            parts.append(sup.cpu())
        origins.append(ori.cpu());refs.append(torch.cat(parts,1))
    ori,sup=torch.cat(origins),torch.cat(refs)
    agg=sup.median(1).values
    z=(1-model.alpha)*ori+model.alpha*agg
    cycles=(z*data['scale']+data['mean']).exp()
    return dict(prediction_cycles=cycles,truth_cycles=data['test']['label'],prediction=z,
                label_mean=data['mean'],label_sample_std=data['scale'],train_ids=data['train']['ids'],test_ids=data['test']['ids'],
                diagnostics=dict(y_ori=ori,y_sup=sup,y_sup_agg=agg,support_index=indices),
                diagnostic_units='standardized_natural_log',scores=common.scores(cycles,data['test']['label']))


def evaluate_run(data,cfg,seed,folder,binding,indices,device):
    final=folder/'final.pt'
    cp=common.checked_checkpoint(final,binding)
    if cp['epoch']!=cfg['model']['epochs']:
        raise ValueError('测试权重未完成固定轮数。')
    sha=common.digest(final);path=folder/'test.pt'
    if path.exists():
        result=common.load(path)
    else:
        historical20.set_seed(seed)
        model=common.MODELS.build(cfg['model'],seed=seed).to(device)
        model.load_state_dict(cp['state'],strict=True)
        result=predict(model,data,indices,device,cfg['runner']['pair_chunk'])
        result.update(binding=binding,checkpoint_sha256=sha)
        common.save(result,path)
    if result['binding']!=binding or result['checkpoint_sha256']!=sha:
        raise ValueError('结果与权重、运行来源绑定错误。')
    if result['train_ids']!=data['train']['ids'] or result['test_ids']!=data['test']['ids']:
        raise ValueError('预测电池名单变化。')
    if not torch.equal(result['truth_cycles'],data['test']['label']) or not torch.equal(result['diagnostics']['support_index'],indices):
        raise ValueError('预测真值或参考被修改。')
    agg=result['diagnostics']['y_sup'].median(1).values
    z=(1-cfg['model']['alpha'])*result['diagnostics']['y_ori']+cfg['model']['alpha']*agg
    torch.testing.assert_close(result['diagnostics']['y_sup_agg'],agg)
    torch.testing.assert_close(result['prediction'],z)
    torch.testing.assert_close(result['prediction_cycles'],(z*data['scale']+data['mean']).exp())
    measured=common.scores(result['prediction_cycles'],result['truth_cycles'])
    if measured!=result['scores']:
        raise ValueError('保存指标不能复算。')
    common.atomic_json(dict(status='complete',binding=binding,final_sha256=sha,test_sha256=common.digest(path),scores=measured),folder/'run.json')
    return dict(task=binding['task'],model=MODEL,seed=seed,**measured)


def preflight(data,cfg,device):
    historical20.set_seed(0)
    model=common.MODELS.build(cfg['model'],seed=0).to(device)
    ds=prepared(model,data['train'],device)
    n=min(cfg['runner']['micro_batch_size'],len(ds))
    sx,sy=model.get_support_set(ds.feature[:n],ds.feature,ds.label,support_is_prepared=True)
    loss=model(ds.feature[:n],ds.label[:n],sx,sy,return_loss=True)
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('训练前向或梯度检查失败。')
    toy=dict(data,test={k:v[:1] for k,v in data['train'].items()})
    ix=torch.arange(32).reshape(1,32)%len(ds)
    output=predict(model,toy,ix,device,cfg['runner']['pair_chunk'])
    return dict(train_count=len(ds),test_count=len(data['test']['ids']),parameters=sum(p.numel() for p in model.parameters()),
                finite=True,loss=float(loss.detach()),training_target_prediction=output['prediction_cycles'].tolist(),
                cuda_peak_GiB=torch.cuda.max_memory_allocated(device)/2**30 if device.startswith('cuda') else 0.)


def write_summary(rows,workspace):
    common.atomic_json(rows,workspace/'per_seed.json')
    import csv
    with (workspace/'per_seed.csv.tmp').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['task','model','seed','RMSE','MAE','MAPE','ACC15'])
        writer.writeheader();writer.writerows(rows)
    os.replace(workspace/'per_seed.csv.tmp',workspace/'per_seed.csv')
    summary=[]
    for task in TASKS:
        selected=[r for r in rows if r['task']==task]
        if len(selected)==8 and {r['seed'] for r in selected}==set(range(8)):
            summary.append(dict(task=task,model=MODEL,seeds=8,comparison='historical_reference_only',
                metrics={k:dict(mean=statistics.mean(r[k] for r in selected),sample_std=statistics.stdev(r[k] for r in selected)) for k in ('RMSE','MAE','MAPE','ACC15')}))
    common.atomic_json(summary,workspace/'summary.json')
    reference_summary=[]
    for task in TASKS:
        path=workspace/'datasets'/task/'additional_historical_per_seed.json'
        prior=json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
        for name in sorted({r['model'] for r in prior}):
            subset=[r for r in prior if r['model']==name]
            if len(subset)==8:
                reference_summary.append(dict(task=task,model=name,seeds=8,comparison='historical_reference_only',
                    metrics={k:dict(mean=statistics.mean(r[k] for r in subset),sample_std=statistics.stdev(r[k] for r in subset)) for k in ('RMSE','MAE','MAPE','ACC15')}))
    common.atomic_json(reference_summary,workspace/'historical_reference_summary.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-root',type=Path,default=Path('/root/autodl-tmp/BatLiNet-main3'))
    parser.add_argument('--original-root',type=Path,default=Path('/root/autodl-tmp/BatLiNet-main2'))
    parser.add_argument('--workspace',type=Path,default=ROOT/'workspaces/mix_cycle_soft_difference_v1')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();torch.set_num_threads(8)
    workspace=args.workspace.resolve()
    with common.queue_lock(workspace):
        try:
            old_queue=ROOT/'workspaces/three_tasks_three_models_v1/.queue.lock'
            if old_queue.exists():
                with common.queue_lock(old_queue.parent):
                    pass  # Refuse GPU contention with the postponed three-task queue.
            if not args.device.startswith('cuda') or not torch.cuda.is_available():
                raise RuntimeError('正式队列要求CUDA，不自动改为CPU。')
            historical20.set_seed(0)
            configs={task:model_config(task) for task in TASKS}
            revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
            identity=dict(git_commit=revision,source_code=sources(),runtime=runtime(args.device),configs=configs,
                          history_root=str(args.history_root.resolve()),original_root=str(args.original_root.resolve()),runs=16)
            if (workspace/'queue.json').exists() and json.loads((workspace/'queue.json').read_text(encoding='utf-8'))!=identity:
                raise ValueError('已有队列代码、提交、环境、配置或来源不同，拒绝混用。')
            common.atomic_json(identity,workspace/'queue.json')
            common.atomic_json(dict(status='preparing',pid=os.getpid(),runs=16),workspace/'status.json')
            shas,gates={},{}
            for task in TASKS:
                data,_,shas[task]=prepare_history(task,args.history_root.resolve(),args.original_root.resolve(),workspace)
                existing_references(task,data,workspace)
                torch.cuda.reset_peak_memory_stats(args.device)
                gates[task]=preflight(data,configs[task],args.device)
                del data;torch.cuda.empty_cache()
            common.atomic_json(gates,workspace/'preflight.json')
            if args.prepare_only:
                common.atomic_json(dict(status='prepared',formal_training_started=False),workspace/'status.json')
                return
            rows=[]
            print('预检通过：仅组合版，先MIX-20后MIX-100，八种子各1000轮，共16次。',flush=True)
            for task in TASKS:
                data=common.load(workspace/'datasets'/task/'data.pt');cfg=configs[task];bindings={}
                for seed in range(8):
                    binding=dict(task=task,model=MODEL,seed=seed,config=cfg,data_sha256=shas[task],
                                 queue=identity,protocol_sha256=common.digest(workspace/'protocols'/task/f'seed_{seed}.pt'))
                    bindings[seed]=binding
                    torch.cuda.reset_peak_memory_stats(args.device)
                    train_run(data,cfg,seed,workspace/task/f'{MODEL}_seed{seed}',binding,args.device)
                    torch.cuda.empty_cache()
                for seed in range(8):
                    indices=common.load(workspace/'protocols'/task/f'seed_{seed}.pt')['indices']
                    row=evaluate_run(data,cfg,seed,workspace/task/f'{MODEL}_seed{seed}',bindings[seed],indices,args.device)
                    rows.append(row);write_summary(rows,workspace)
                    print(json.dumps(row,ensure_ascii=False),flush=True)
                    torch.cuda.empty_cache()
                del data
            if len(rows)!=16:
                raise ValueError('结果数不完整。')
            common.atomic_json(dict(status='complete',training_runs=16,predictions=16),workspace/'status.json')
            print('16次组合训练及最终预测核验全部完成。汇总：'+str(workspace/'summary.json'),flush=True)
        except BaseException as error:
            common.atomic_json(dict(status='failed',error=repr(error),traceback=traceback.format_exc()),workspace/'status.json')
            raise


if __name__=='__main__':main()
