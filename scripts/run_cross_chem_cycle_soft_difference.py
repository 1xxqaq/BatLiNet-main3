"""Only the existing combination: LFP-source few-label cross-chemistry transfer."""
import argparse
import contextlib
import csv
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_three_tasks as common
from scripts import transfer_cycle_soft_data as data
from src.models.rul_predictors import cycle_mixer_soft_aligned_difference_batlinet

MODEL='latent_cycle_soft_difference'
CONFIG=ROOT/'configs/cross_chem_cycle_soft_difference_v1/combination.yaml'


def read_config():
    cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    expected=dict(schema=1,maximum_epochs=100,patience=5,logical_batch=128,micro_batch=8,
        inference_chunk=8,weight_decay=.01,shuffle=False,precision='float32',checkpoint_every=1,
        selection='first_lowest_validation_relation_mse',initial_weights='fresh_random_no_MIX_pretraining',
        training_pairs='exhaustive_target_times_LFP_80_percent',
        validation_pairs='complementary_20_percent_same_training_cells',evaluation_references='all_275_LFP',
        test_during_training=False,label_space='natural_log_common_zscore')
    if cfg['runner']!=expected:raise ValueError('队列配置偏离冻结迁移协议。')
    expected_model=dict(name='CycleMixerSoftAlignedDifferenceBatLiNetRULPredictor',in_channels=6,channels=32,
        input_height=100,input_width=1000,epochs=100,train_batch_size=128,test_batch_size=1,
        train_support_size=1,test_support_size=275,gradient_accumulation_steps=1,evaluate_freq=100,
        checkpoint_freq=1,filter_cycles=False,attention_channels=64,attention_heads=4,attention_layers=1,
        attention_dropout=.1,attention_mlp_ratio=2,head_hidden_channels=64,encoder_dropout=.1,
        alpha=.5,lr=.001,cycle_mixer_hidden=16,workspace=None)
    if cfg['model']!=expected_model:raise ValueError('组合结构或正式训练参数偏离本次冻结配置。')
    return cfg


def sources():
    paths=[Path(__file__),ROOT/'scripts/transfer_cycle_soft_data.py',ROOT/'scripts/run_three_tasks.py',CONFIG]
    paths+=list((ROOT/'src').rglob('*.py'))
    # Record local author artifacts actually inspected, not an assumed remote version.
    paths += [ROOT/'transfer_reproduce/preprocess_feature.ipynb',ROOT/'transfer_reproduce/finalized.ipynb',ROOT/'transfer_exp.ipynb']
    return {p.relative_to(ROOT).as_posix():common.digest(p) for p in sorted(paths)}


def make_model(cfg,seed,device):
    common.seed_all(seed)
    return common.MODELS.build(dict(cfg['model']),seed=seed).to(device)


def add_z(part,mean,scale):
    part=dict(part);part['z']=(part['label'].log()-mean)/scale
    if not torch.isfinite(part['z']).all():raise ValueError('标签标准化非有限。')
    return part


def pair_components(model,source,target,pairs,device):
    n=len(source['label']);ti=pairs//n;si=pairs%n
    x=target['feature'][ti].to(device);y=target['z'][ti].to(device)
    sx=source['feature'][si].to(device).unsqueeze(1)
    sy=source['z'][si].to(device).unsqueeze(1)
    own,sup,_,_,_=model.compute_prediction_components(x,sx,sy)
    return own,sup[:,0],y


def train_epoch(model,optimizer,source,target,pairs,cfg,device):
    model.train();logical=cfg['runner']['logical_batch'];micro=cfg['runner']['micro_batch']
    total=0.;steps=0
    for start in range(0,len(pairs),logical):
        block=pairs[start:start+logical];optimizer.zero_grad(set_to_none=True)
        for offset in range(0,len(block),micro):
            chosen=block[offset:offset+micro]
            own,sup,y=pair_components(model,source,target,chosen,device)
            loss=(1-model.alpha)*((own-y)**2).mean()+model.alpha*((sup-y)**2).mean()
            if not torch.isfinite(loss):raise FloatingPointError('电池对训练损失非有限。')
            (loss*len(chosen)/len(block)).backward()
            total+=float(loss.detach())*len(chosen)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('电池对梯度非有限。')
        optimizer.step();steps+=1
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('优化器更新产生非有限参数。')
    return dict(train_loss=total/len(pairs),optimizer_steps=steps)


@torch.no_grad()
def validation_loss(model,source,target,pairs,micro,device):
    model.eval();total=0.
    for start in range(0,len(pairs),micro):
        chosen=pairs[start:start+micro]
        _,sup,y=pair_components(model,source,target,chosen,device)
        loss=((sup-y)**2).sum()
        if not torch.isfinite(loss):raise FloatingPointError('验证关系误差非有限。')
        total+=float(loss)
    return total/len(pairs)


def check_checkpoint(path,binding):
    state=common.load(path)
    if state['binding']!=binding:raise ValueError(f'权重/恢复文件绑定不同：{path}')
    return state


def train_run(source,target,p,cfg,folder,binding,device):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    model=make_model(cfg,p['seed'],device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['model']['lr'],weight_decay=cfg['runner']['weight_decay'])
    rng_generator=torch.Generator().manual_seed(p['seed'])
    final=folder/'final.pt';progress=folder/'progress.pt';receipt=folder/'run.json'
    if receipt.exists() and json.loads(receipt.read_text(encoding='utf-8'))['binding']!=binding:
        raise ValueError('已有运行记录绑定不同。')
    if final.exists():
        state=check_checkpoint(final,binding)
        if not 1<=state['selected_epoch']<=state['trained_epochs']<=cfg['runner']['maximum_epochs']:
            raise ValueError('已保存最佳权重轮次错误。')
        common.align_log(folder,state['trained_epochs'])
        if receipt.exists():
            r=json.loads(receipt.read_text(encoding='utf-8'))
            if 'final_sha256' in r and r['final_sha256']!=common.digest(final):raise ValueError('最佳权重指纹变化。')
        return state
    start=0;best_loss=None;best_epoch=0;best_state=None;bad_epochs=0
    if progress.exists():
        state=check_checkpoint(progress,binding)
        start=state['epoch'];best_loss=state['best_loss'];best_epoch=state['best_epoch']
        best_state=state['best_state'];bad_epochs=state['bad_epochs']
        if (not 0<=start<=cfg['runner']['maximum_epochs'] or not 0<=best_epoch<=start
                or not 0<=bad_epochs<=cfg['runner']['patience']
                or (start>0 and (best_state is None or best_epoch<1 or best_loss is None or not math.isfinite(best_loss)))):
            raise ValueError('恢复检查点的轮次或早停状态不符合协议。')
        model.load_state_dict(state['state'],strict=True);optimizer.load_state_dict(state['optimizer'])
        common.restore_rng(state['rng'],rng_generator,device)
    common.align_log(folder,start)
    common.atomic_json(dict(status='training',binding=binding),receipt)
    if start==0:
        common.save(dict(epoch=0,binding=binding,state=model.state_dict(),optimizer=optimizer.state_dict(),
            best_state=None,best_epoch=0,best_loss=None,bad_epochs=0,rng=common.rng_state(rng_generator,device)),progress)
    epoch=start
    for epoch in range(start+1,cfg['runner']['maximum_epochs']+1):
        if bad_epochs>=cfg['runner']['patience']:epoch-=1;break
        begin=time.monotonic()
        measured=train_epoch(model,optimizer,source,target,p['train_pairs'],cfg,device)
        val=validation_loss(model,source,target,p['val_pairs'],cfg['runner']['micro_batch'],device)
        if best_loss is None or val<best_loss:
            best_loss=val;best_epoch=epoch;best_state=common.cpu(model.state_dict());bad_epochs=0
            # cpu() alone can alias tensors when the model itself is on CPU.
            best_state={k:v.clone() for k,v in best_state.items()}
        else:bad_epochs+=1
        record=dict(epoch=epoch,**measured,validation_relation_mse=val,best_epoch=best_epoch,
            best_validation_relation_mse=best_loss,bad_epochs=bad_epochs,seconds=time.monotonic()-begin)
        with (folder/'train.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        common.save(dict(epoch=epoch,binding=binding,state=model.state_dict(),optimizer=optimizer.state_dict(),
            best_state=best_state,best_epoch=best_epoch,best_loss=best_loss,bad_epochs=bad_epochs,
            rng=common.rng_state(rng_generator,device)),progress)
        common.atomic_json(dict(status='training',target=p['target_chemistry'],train_count=p['target_train_count'],
            seed=p['seed'],epoch=epoch,best_epoch=best_epoch),folder.parents[2]/'status.json')
        print(f"{p['target_chemistry']} 训练{p['target_train_count']}块 种子{p['seed']} 轮次{epoch} 验证关系误差{val:.6f}",flush=True)
    if best_state is None:raise ValueError('未获得有效验证权重。')
    state=dict(binding=binding,state=best_state,selected_epoch=best_epoch,trained_epochs=epoch,
               validation_relation_mse=best_loss,selection=cfg['runner']['selection'])
    common.save(state,final)
    common.atomic_json(dict(status='trained',binding=binding,selected_epoch=best_epoch,trained_epochs=epoch,
        validation_relation_mse=best_loss,final_sha256=common.digest(final)),receipt)
    return state


@torch.no_grad()
def predict(model,source,target,mean,scale,chunk,device):
    model.eval()
    # Cache reference encodings only for evaluation, after the selected weights are loaded.
    tokens=torch.cat([model.cell_encoder(source['feature'][i:i+chunk].to(device))
                      for i in range(0,len(source['label']),chunk)])
    z_source=source['z'].to(device);own_rows=[];sup_rows=[]
    for start in range(len(target['label'])):
        query=model.cell_encoder(target['feature'][start:start+1].to(device))
        own=model.ori_head(query).reshape(())
        values=[]
        for i in range(0,len(tokens),chunk):
            ref=tokens[i:i+chunk]
            relation=query.expand(len(ref),-1,-1)
            for block in model.cross_attention:relation=block(relation,ref)
            values.append(model.support_head(relation).reshape(-1)+z_source[i:i+len(ref)])
        own_rows.append(own.cpu());sup_rows.append(torch.cat(values).cpu())
    y_ori=torch.stack(own_rows);y_sup=torch.stack(sup_rows);agg=y_sup.median(1).values
    prediction_z=model.combine_predictions(y_ori,agg)
    prediction_cycles=(prediction_z*scale+mean).exp()
    if not torch.isfinite(prediction_cycles).all():raise FloatingPointError('最终寿命预测非有限，不裁剪或剔除。')
    return dict(prediction=prediction_z,prediction_cycles=prediction_cycles,truth_cycles=target['label'],
        label_mean=mean,label_sample_std=scale,source_ids=source['ids'],target_test_ids=target['ids'],
        diagnostics=dict(y_ori=y_ori,y_sup=y_sup,y_sup_agg=agg,
            support_index=torch.arange(len(source['ids'])).expand(len(target['ids']),-1).clone()),
        diagnostic_units='standardized_natural_log')


def evaluate(source,target_test,p,cfg,folder,binding,mean,scale,device):
    folder=Path(folder);final=folder/'final.pt';test=folder/'test.pt';receipt=folder/'run.json'
    state=check_checkpoint(final,binding);weight_sha=common.digest(final)
    if test.exists():
        output=common.load(test)
        if output['binding']!=binding or output['checkpoint_sha256']!=weight_sha:
            raise ValueError('已有预测与本次权重/运行绑定不同。')
        r=json.loads(receipt.read_text(encoding='utf-8'))
        if r.get('test_sha256') is not None and r['test_sha256']!=common.digest(test):raise ValueError('预测指纹变化。')
    else:
        model=make_model(cfg,p['seed'],device);model.load_state_dict(state['state'],strict=True)
        output=predict(model,source,target_test,mean,scale,cfg['runner']['inference_chunk'],device)
        output.update(binding=binding,checkpoint_sha256=weight_sha,selected_epoch=state['selected_epoch'],
            trained_epochs=state['trained_epochs'],scores=common.scores(output['prediction_cycles'],output['truth_cycles']))
        common.save(output,test)
    if output['source_ids']!=p['source_ids'] or output['target_test_ids']!=p['target_test_ids']:
        raise ValueError('预测电池ID/参考顺序不同。')
    if (output['selected_epoch']!=state['selected_epoch'] or output['trained_epochs']!=state['trained_epochs']
            or output['diagnostic_units']!='standardized_natural_log'):
        raise ValueError('预测权重轮次或分支单位错误。')
    torch.testing.assert_close(output['truth_cycles'],target_test['label'],rtol=0,atol=0)
    torch.testing.assert_close(output['label_mean'],mean,rtol=0,atol=0)
    torch.testing.assert_close(output['label_sample_std'],scale,rtol=0,atol=0)
    expected_ix=torch.arange(len(p['source_ids'])).expand(len(p['target_test_ids']),-1)
    if not torch.equal(output['diagnostics']['support_index'],expected_ix):raise ValueError('实际参考索引不同。')
    torch.testing.assert_close(output['diagnostics']['y_sup_agg'],output['diagnostics']['y_sup'].median(1).values)
    torch.testing.assert_close(output['prediction'],(1-cfg['model']['alpha'])*output['diagnostics']['y_ori']+cfg['model']['alpha']*output['diagnostics']['y_sup_agg'])
    torch.testing.assert_close(output['prediction_cycles'],(output['prediction']*scale+mean).exp())
    scores=common.scores(output['prediction_cycles'],output['truth_cycles'])
    for k,v in scores.items():
        if not math.isclose(v,output['scores'][k],abs_tol=1e-10,rel_tol=1e-10):raise ValueError('预测分数不能复算。')
    common.atomic_json(dict(status='complete',binding=binding,selected_epoch=state['selected_epoch'],
        trained_epochs=state['trained_epochs'],validation_relation_mse=state['validation_relation_mse'],
        final_sha256=weight_sha,test_sha256=common.digest(test),scores=scores),receipt)
    return dict(target=p['target_chemistry'],train_count=p['target_train_count'],model=MODEL,seed=p['seed'],
        selected_epoch=state['selected_epoch'],trained_epochs=state['trained_epochs'],**scores)


def write_summary(rows,workspace,expected_runs):
    workspace=Path(workspace)
    common.atomic_json(rows,workspace/'per_seed.json')
    if rows:
        temporary=workspace/'per_seed.csv.tmp'
        with temporary.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        os.replace(temporary,workspace/'per_seed.csv')
    common.atomic_json(summarize(rows,expected_runs),workspace/'summary.json')


def summarize(rows,expected_runs):
    summary=[]
    for target in data.TEST_COUNTS:
        for count in data.TRAIN_COUNTS[target]:
            group=[r for r in rows if r['target']==target and r['train_count']==count]
            if len(group)==8 and {r['seed'] for r in group}==set(range(8)):
                summary.append(dict(target=target,train_count=count,model=MODEL,seeds=8,
                    metrics={k:dict(mean=statistics.mean(r[k] for r in group),sample_std=statistics.stdev(r[k] for r in group))
                             for k in ('RMSE','MAE','MAPE','ACC15')}))
    return dict(expected_runs=expected_runs,completed_runs=len(rows),groups=summary)


def completed_rows(workspace, identity):
    """Rebuild summaries from verified completed receipts, including after a restart."""
    workspace=Path(workspace);rows=[]
    for target,count,seed in identity['runs']:
        folder=workspace/target/f'train_{count}'/f'{MODEL}_seed{seed}'
        receipt=folder/'run.json'
        if not receipt.exists():continue
        r=json.loads(receipt.read_text(encoding='utf-8'))
        if r['status']!='complete':continue
        binding=r['binding'];final=folder/'final.pt';test=folder/'test.pt'
        if (binding['queue']!=identity or binding['target']!=target
                or binding['train_count']!=count or binding['seed']!=seed
                or binding['manifest_sha256']!=common.digest(workspace/'dataset/manifest.json')
                or binding['protocol_sha256']!=common.digest(workspace/'protocols'/target/f'train_{count}'/f'seed_{seed}.pt')):
            raise ValueError('汇总中的运行与冻结协议不符。')
        if common.digest(final)!=r['final_sha256'] or common.digest(test)!=r['test_sha256']:
            raise ValueError('已完成权重或预测的指纹变化。')
        state=check_checkpoint(final,binding);output=common.load(test)
        if (output['binding']!=binding or output['checkpoint_sha256']!=r['final_sha256']
                or output['selected_epoch']!=state['selected_epoch']
                or output['trained_epochs']!=state['trained_epochs']):
            raise ValueError('汇总预测与最佳权重不符。')
        measured=common.scores(output['prediction_cycles'],output['truth_cycles'])
        for key,value in measured.items():
            if not math.isclose(value,r['scores'][key],rel_tol=1e-10,abs_tol=1e-10):
                raise ValueError('汇总指标不能从保存预测复算。')
        rows.append(dict(target=target,train_count=count,model=MODEL,seed=seed,
            selected_epoch=state['selected_epoch'],trained_epochs=state['trained_epochs'],**measured))
    return rows


def report(workspace):
    queue=Path(workspace)/'queue.json'
    if queue.exists():
        identity=json.loads(queue.read_text(encoding='utf-8'))
        if identity['model']!=MODEL:raise ValueError('不是本迁移队列。')
        value=summarize(completed_rows(workspace,identity),len(identity['runs']))
    else:
        value=json.loads((Path(workspace)/'summary.json').read_text(encoding='utf-8'))
    print(f"已保存结果：{value['completed_runs']}/{value['expected_runs']}；以下仅显示完整八种子的组。")
    for group in value['groups']:
        print(f"\n{group['target']}，目标训练{group['train_count']}块，组合版八种子")
        for key,label,scale,unit in [('RMSE','均方根误差',1,'循环'),('MAE','平均绝对误差',1,'循环'),
                                   ('MAPE','平均绝对百分比误差',100,'%'),('ACC15','15%以内准确率',100,'%')]:
            m=group['metrics'][key];print(f"  {label}：{m['mean']*scale:.4f} ± {m['sample_std']*scale:.4f} {unit}")


def plan(targets,counts):
    rows=[]
    for target in targets:
        choices=data.TRAIN_COUNTS[target] if counts is None else counts
        for count in choices:
            if count not in data.TRAIN_COUNTS[target]:raise ValueError(f'{target}不能使用{count}个训练电池，固定测试划分后样本不足。')
            rows.extend((target,count,seed) for seed in range(8))
    if len(rows)!=len(set(rows)):raise ValueError('任务重复。')
    return rows


def preflight(source,target,p,cfg,mean,scale,device):
    torch.cuda.reset_peak_memory_stats(device) if device.startswith('cuda') else None
    model=make_model(cfg,0,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['model']['lr'],weight_decay=cfg['runner']['weight_decay'])
    measured=train_epoch(model,optimizer,source,target,p['train_pairs'][:cfg['runner']['logical_batch']],cfg,device)
    val=validation_loss(model,source,target,p['val_pairs'][:16],cfg['runner']['micro_batch'],device)
    toy={k:(v[:1] if torch.is_tensor(v) else v[:1]) for k,v in target.items()}
    prediction=predict(model,source,toy,mean,scale,cfg['runner']['inference_chunk'],device)
    return dict(parameters=sum(t.numel() for t in model.parameters()),training_loss=measured['train_loss'],
        validation_relation_mse=val,evaluation_source_count=len(source['ids']),
        training_target_prediction=prediction['prediction_cycles'].tolist(),
        cuda_peak_GiB=torch.cuda.max_memory_allocated(device)/2**30 if device.startswith('cuda') else None)


def main():
    parser=argparse.ArgumentParser(description='仅组合版的论文数据协议跨化学体系迁移')
    parser.add_argument('--source-root',type=Path,default=Path('/root/autodl-tmp/BatLiNet-main3'))
    parser.add_argument('--workspace',type=Path,default=ROOT/'workspaces/cross_chem_cycle_soft_difference_v1')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--targets',nargs='+',choices=list(data.TEST_COUNTS),default=list(data.TEST_COUNTS))
    parser.add_argument('--counts',nargs='+',type=int)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--report',action='store_true')
    args=parser.parse_args();workspace=args.workspace.resolve()
    if args.report:report(workspace);return
    if os.name!='posix' or not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('正式迁移队列仅在Linux CUDA服务器执行，本地只运行合成测试。')
    cfg=read_config();jobs=plan(args.targets,args.counts)
    identity=dict(schema=1,model=MODEL,policy=data.POLICY,config=cfg,runs=[list(j) for j in jobs],
        source_root=str(args.source_root.resolve()),code_hashes=sources(),runtime=common.runtime(args.device),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    with contextlib.ExitStack() as stack:
        stack.enter_context(common.queue_lock(workspace))
        for name in ('mix_cycle_soft_difference_v1','three_tasks_three_models_v1'):
            old=ROOT/'workspaces'/name
            if old.exists():stack.enter_context(common.queue_lock(old))
        try:
            queue=workspace/'queue.json'
            if queue.exists() and json.loads(queue.read_text(encoding='utf-8'))!=identity:
                raise ValueError('已有队列绑定的代码、配置、任务或环境不同；不可混合恢复。')
            common.atomic_json(identity,queue)
            common.atomic_json(dict(status='preparing',pid=os.getpid(),runs=len(jobs)),workspace/'status.json')
            manifest=data.prepare(args.source_root,workspace,identity['code_hashes'])
            manifest_sha=common.digest(workspace/'dataset/manifest.json')
            protocols={}
            for target,count,seed in jobs:
                p,path=data.protocol(manifest,target,count,seed,workspace);protocols[(target,count,seed)]=(p,path)
            source=data.load_part(sorted(r['cell_id'] for r in manifest['records'] if r['chemistry']=='LFP'),manifest,workspace)
            target,count,seed=max(jobs,key=lambda j:j[1]);p,_=protocols[(target,count,seed)]
            training=data.load_part(p['target_train_ids'],manifest,workspace);mean,scale=data.label_statistics(source,training)
            gate=preflight(add_z(source,mean,scale),add_z(training,mean,scale),p,cfg,mean,scale,args.device)
            common.atomic_json(dict(observed_counts=manifest['observed_counts'],runs=len(jobs),**gate),workspace/'preflight.json')
            del training;torch.cuda.empty_cache()
            if args.prepare_only:
                common.atomic_json(dict(status='prepared',formal_training_started=False),workspace/'status.json');return
            rows=completed_rows(workspace,identity);write_summary(rows,workspace,len(jobs))
            print(f'预检通过：仅组合版，共{len(jobs)}次，最多100轮、验证关系误差早停，不使用测试选权重。',flush=True)
            for target,count,seed in jobs:
                p,path=protocols[(target,count,seed)]
                training=data.load_part(p['target_train_ids'],manifest,workspace)
                mean,scale=data.label_statistics(source,training)
                src=add_z(source,mean,scale);tr=add_z(training,mean,scale)
                binding=dict(target=target,train_count=count,seed=seed,model=MODEL,queue=identity,
                    manifest_sha256=manifest_sha,protocol_sha256=common.digest(path),
                    config=cfg,label_mean=float(mean),label_sample_std=float(scale))
                folder=workspace/target/f'train_{count}'/f'{MODEL}_seed{seed}'
                train_run(src,tr,p,cfg,folder,binding,args.device)
                # Held-out inputs are passed to evaluation only after weight selection;
                # preprocessing receipts contain their labels, but fitting never uses them.
                testing=data.load_part(p['target_test_ids'],manifest,workspace)
                row=evaluate(src,testing,p,cfg,folder,binding,mean,scale,args.device)
                rows=[r for r in rows if (r['target'],r['train_count'],r['seed'])!=(target,count,seed)]+[row]
                order={job:i for i,job in enumerate(jobs)}
                rows.sort(key=lambda r:order[(r['target'],r['train_count'],r['seed'])])
                write_summary(rows,workspace,len(jobs))
                common.atomic_json(dict(status='running',completed_runs=len(rows),total_runs=len(jobs)),workspace/'status.json')
                print(json.dumps(row,ensure_ascii=False),flush=True)
                del training,tr,src,testing;torch.cuda.empty_cache()
            common.atomic_json(dict(status='complete',training_runs=len(rows),predictions=len(rows)),workspace/'status.json')
            report(workspace)
        except BaseException as error:
            common.atomic_json(dict(status='failed',error=repr(error),traceback=traceback.format_exc()),workspace/'status.json')
            raise


if __name__=='__main__':main()