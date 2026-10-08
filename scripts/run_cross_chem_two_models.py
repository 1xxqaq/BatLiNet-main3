"""Same-split original author transfer and unchanged combination; 224 model cases."""
import argparse
import contextlib
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import traceback
from pathlib import Path

import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_three_tasks as common
from scripts import run_cross_chem_cycle_soft_difference as combo
from scripts import original_cross_chem_transfer as original
from scripts import transfer_cycle_soft_data as data

MODELS=(original.MODEL,combo.MODEL)
CONFIG=ROOT/'configs/cross_chem_two_models_v2/original.yaml'
SHARED_POLICY={k:v for k,v in data.POLICY.items() if k not in
    ('feature_clip','label_space','label_statistics','selection')}
SHARED_POLICY.update(version=2,model_training_rules='separate_frozen_model_configs',
    stored_features='raw_unclipped_author_features',original_target_test_ids_available=False)


def read_configs():
    cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    expected=dict(schema=1,name=original.MODEL,
        network=dict(in_channels=6,channels=32,input_height=100,input_width=1000,kernel_size=3,act_fn='relu'),
        preprocessing=dict(cycle_diff_base=9,clip_absolute_over=10.,clip_after_difference=True),
        training=dict(own_epochs=100,relation_epochs=100,relation_patience=5,own_batch=32,relation_batch=128,
            micro_batch=8,inference_chunk=8,lr=.001,weight_decay=.01,shuffle=False,precision='float32',checkpoint_every=1),
        label=dict(units='raw_cycles',own_statistics='target_training_sample_std_else_LFP_sample_std',
            delta_statistics='training_pairs_only_sample_std'),
        prediction=dict(reference_count=275,aggregation='positive_prediction_median_in_cycles',
            fusion='arithmetic_mean_own_relation',shared_output_layer=False,active_dropout=False,test_during_training=False))
    if cfg!=expected:raise ValueError('原始迁移配置偏离冻结协议。')
    return {original.MODEL:cfg,combo.MODEL:combo.read_config()}


def plan(targets,counts):
    return [(target,n,seed,model) for target,n,seed in combo.plan(targets,counts) for model in MODELS]


def code_hashes():
    result=combo.sources()
    for path in (Path(__file__),ROOT/'scripts/original_cross_chem_transfer.py',CONFIG):
        result[path.relative_to(ROOT).as_posix()]=common.digest(path)
    return result


def clip_part(part):
    part=dict(part);x=part['feature'].clone();x[x.abs()>10]=0;part['feature']=x
    return part


def summarize(rows,expected):
    groups=[]
    for target in data.TEST_COUNTS:
        for count in data.TRAIN_COUNTS[target]:
            for model in MODELS:
                group=[row for row in rows if (row['target'],row['train_count'],row['model'])==(target,count,model)]
                if len(group)!=8 or {row['seed'] for row in group}!=set(range(8)):continue
                groups.append(dict(target=target,train_count=count,model=model,seeds=8,
                    metrics={key:dict(mean=statistics.mean(row[key] for row in group),
                        sample_std=statistics.stdev(row[key] for row in group),
                        population_std=statistics.pstdev(row[key] for row in group)) for key in ('RMSE','MAE','MAPE','ACC15')}))
    differences=[]
    lookup={(row['target'],row['train_count'],row['seed'],row['model']):row for row in rows}
    for target in data.TEST_COUNTS:
        for count in data.TRAIN_COUNTS[target]:
            pairs=[(lookup.get((target,count,seed,original.MODEL)),lookup.get((target,count,seed,combo.MODEL))) for seed in range(8)]
            if any(a is None or b is None for a,b in pairs):continue
            differences.append(dict(target=target,train_count=count,seeds=8,
                positive_means_combination_better=True,metrics={key:dict(
                    mean_advantage=statistics.mean((a[key]-b[key])*(1 if key!='ACC15' else -1) for a,b in pairs),
                    better_seeds=sum(b[key]<a[key] if key!='ACC15' else b[key]>a[key] for a,b in pairs))
                    for key in ('RMSE','MAE','MAPE','ACC15')}))
    return dict(expected_runs=expected,completed_runs=len(rows),groups=groups,paired_differences=differences,
        paper_reference=dict(url='https://www.nature.com/articles/s42256-024-00972-x/figures/4',
            metric='MAPE',numerical_values_available=False,comparison_status='待实测完成及取得论文精确数值；不估造数值'))


def write_summary(rows,workspace,expected):
    common.atomic_json(rows,workspace/'per_seed.json');common.atomic_json(summarize(rows,expected),workspace/'summary.json')
    if rows:
        temporary=workspace/'per_seed.csv.tmp'
        with temporary.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        os.replace(temporary,workspace/'per_seed.csv')


def completed_rows(workspace,identity):
    workspace=Path(workspace);rows=[]
    for target,count,seed,model in identity['runs']:
        folder=workspace/target/f'train_{count}'/f'{model}_seed{seed}';receipt=folder/'run.json'
        if not receipt.exists():continue
        r=json.loads(receipt.read_text(encoding='utf-8'))
        if r['status']!='complete':continue
        binding=r['binding'];final=folder/'final.pt';test=folder/'test.pt'
        ppath=workspace/'protocols'/target/f'train_{count}'/f'seed_{seed}.pt'
        if (binding['queue']!=identity or binding['model']!=model or binding['target']!=target
                or binding['train_count']!=count or binding['seed']!=seed
                or binding['manifest_sha256']!=common.digest(workspace/'dataset/manifest.json')
                or binding['protocol_sha256']!=common.digest(ppath)):
            raise ValueError('双模型汇总与数据或协议绑定不同。')
        if common.digest(final)!=r['final_sha256'] or common.digest(test)!=r['test_sha256']:
            raise ValueError('双模型预测或权重指纹变化。')
        saved=common.load(final);output=common.load(test);p=common.load(ppath)
        if (saved['binding']!=binding or output['binding']!=binding
                or output['checkpoint_sha256']!=r['final_sha256']
                or output['source_ids']!=p['source_ids'] or output['target_test_ids']!=p['target_test_ids']):
            raise ValueError('双模型测试绑定或电池顺序错误。')
        for key in ('selected_epoch','trained_epochs'):
            if output[key]!=saved[key]:raise ValueError('双模型预测轮次与权重不同。')
        scores=common.scores(output['prediction_cycles'],output['truth_cycles'])
        for key,value in scores.items():
            if not math.isclose(value,r['scores'][key],rel_tol=1e-10,abs_tol=1e-10):raise ValueError('双模型汇总不能复算。')
        rows.append(dict(target=target,train_count=count,model=model,seed=seed,
            selected_epoch=saved['selected_epoch'],trained_epochs=saved['trained_epochs'],**scores))
    return rows


def report(workspace):
    workspace=Path(workspace);identity=json.loads((workspace/'queue.json').read_text(encoding='utf-8'))
    value=summarize(completed_rows(workspace,identity),len(identity['runs']))
    print(f"已保存：{value['completed_runs']}/{value['expected_runs']}；均值 ± 样本标准差，以下只显示完整八种子组。")
    labels={original.MODEL:'原始BatLiNet（作者本地迁移实现复跑）',combo.MODEL:'循环轴残差＋软匹配差分组合'}
    for group in value['groups']:
        print(f"\n{group['target']} 目标训练{group['train_count']}块：{labels[group['model']]}")
        for key,name,mult,unit in [('RMSE','均方根误差',1,'循环'),('MAE','平均绝对误差',1,'循环'),
            ('MAPE','平均绝对百分比误差',100,'%'),('ACC15','15%以内准确率',100,'%')]:
            m=group['metrics'][key];print(f"  {name}：{m['mean']*mult:.4f} ± {m['sample_std']*mult:.4f} {unit}")
        print(f"  对照作者图表口径的MAPE总体标准差：{group['metrics']['MAPE']['population_std']*100:.4f} %")
    for item in value['paired_differences']:
        m=item['metrics']['MAPE'];print(f"{item['target']}训练{item['train_count']}块，组合相对原始MAPE平均优势{m['mean_advantage']*100:.4f}个百分点，更好种子{m['better_seeds']}/8")
    print('与论文比较对应图4的MAPE；本地作者实现、固定划分及修正差异已记录，尚无论文精确数值自动差距表。')


def preflight(raw_source,raw_training,p,configs,device):
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
    cfg=configs[original.MODEL];stats=original.statistics(raw_source,raw_training,p)
    branches=original.make_branches(cfg,0,device);measure={}
    for stage,indices in [('relation',p['train_pairs'][:128]),('own',torch.arange(len(raw_training['label'])) )]:
        optimizer=torch.optim.AdamW(branches[stage].parameters(),lr=.001,weight_decay=.01)
        measure[stage]=original.train_epoch(branches[stage],raw_source,raw_training,indices,stage,stats,cfg,optimizer,device)
    measure['validation_relation_mse']=original.validate(branches['relation'],raw_source,raw_training,p['val_pairs'][:16],stats,cfg,device)
    small={key:value[:1] for key,value in raw_training.items()}
    output=original.predict(branches,raw_source,small,stats,cfg,device)
    if not output['valid']:raise FloatingPointError('原始模型预检无正参考预测。')
    measure.update(parameters=sum(t.numel() for m in branches.values() for t in m.parameters()),
                   reference_count=len(raw_source['ids']),
                   cuda_peak_GiB=torch.cuda.max_memory_allocated(device)/2**30 if device.startswith('cuda') else None)
    del branches,optimizer;torch.cuda.empty_cache() if device.startswith('cuda') else None
    clipped_source=clip_part(raw_source);clipped_training=clip_part(raw_training)
    mean,scale=data.label_statistics(clipped_source,clipped_training)
    gate=combo.preflight(combo.add_z(clipped_source,mean,scale),combo.add_z(clipped_training,mean,scale),
        p,configs[combo.MODEL],mean,scale,device)
    return {original.MODEL:measure,combo.MODEL:gate}


def main():
    parser=argparse.ArgumentParser(description='原始BatLiNet迁移与组合版同划分合并队列')
    parser.add_argument('--source-root',type=Path,default=Path('/root/autodl-tmp/BatLiNet-main3'))
    parser.add_argument('--workspace',type=Path,default=ROOT/'workspaces/cross_chem_two_models_v2')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--targets',nargs='+',choices=list(data.TEST_COUNTS),default=list(data.TEST_COUNTS))
    parser.add_argument('--counts',nargs='+',type=int)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--report',action='store_true')
    args=parser.parse_args();workspace=args.workspace.resolve()
    if args.report:report(workspace);return
    if os.name!='posix' or not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('正式迁移仅在Linux CUDA服务器运行，本地只做合成检查。')
    configs=read_configs();jobs=plan(args.targets,args.counts)
    identity=dict(schema=2,models=list(MODELS),data_policy=SHARED_POLICY,configs=configs,
        runs=[list(j) for j in jobs],source_root=str(args.source_root.resolve()),code_hashes=code_hashes(),
        runtime=common.runtime(args.device),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    with contextlib.ExitStack() as stack:
        stack.enter_context(common.queue_lock(workspace))
        for name in ('cross_chem_cycle_soft_difference_v1','mix_cycle_soft_difference_v1','three_tasks_three_models_v1'):
            other=ROOT/'workspaces'/name
            if other.exists():stack.enter_context(common.queue_lock(other))
        try:
            queue=workspace/'queue.json'
            if queue.exists() and json.loads(queue.read_text(encoding='utf-8'))!=identity:raise ValueError('双模型队列身份变化，不能混合恢复。')
            common.atomic_json(identity,queue);common.atomic_json(dict(status='preparing',total_runs=len(jobs)),workspace/'status.json')
            manifest=data.prepare(args.source_root,workspace,identity['code_hashes'],SHARED_POLICY)
            protocols={}
            for target,count,seed,_ in jobs:
                key=(target,count,seed)
                if key not in protocols:protocols[key]=data.protocol(manifest,target,count,seed,workspace,SHARED_POLICY)
            source_ids=sorted(r['cell_id'] for r in manifest['records'] if r['chemistry']=='LFP')
            raw_source=data.load_part(source_ids,manifest,workspace,clip=False)
            target,count,seed,_=max(jobs,key=lambda j:j[1]);p,_=protocols[(target,count,seed)]
            raw_training=data.load_part(p['target_train_ids'],manifest,workspace,clip=False)
            gate=preflight(raw_source,raw_training,p,configs,args.device)
            common.atomic_json(dict(observed_counts=manifest['observed_counts'],total_runs=len(jobs),models=gate),workspace/'preflight.json')
            del raw_training;torch.cuda.empty_cache()
            if args.prepare_only:
                common.atomic_json(dict(status='prepared',formal_training_started=False),workspace/'status.json');return
            clipped_source=clip_part(raw_source);rows=completed_rows(workspace,identity)
            write_summary(rows,workspace,len(jobs));order={j:i for i,j in enumerate(jobs)}
            print(f'两模型预检通过，共{len(jobs)}组；固定同一目标名单与配对划分，逐组顺序执行。',flush=True)
            for target,count,seed,model in jobs:
                p,path=protocols[(target,count,seed)];cfg=configs[model]
                folder=workspace/target/f'train_{count}'/f'{model}_seed{seed}'
                binding=dict(queue=identity,model=model,target=target,train_count=count,seed=seed,config=cfg,
                    manifest_sha256=common.digest(workspace/'dataset/manifest.json'),protocol_sha256=common.digest(path))
                common.atomic_json(dict(status='running',model=model,target=target,train_count=count,seed=seed,
                    completed_runs=len(rows),total_runs=len(jobs)),workspace/'status.json')
                training=data.load_part(p['target_train_ids'],manifest,workspace,clip=False)
                if model==original.MODEL:
                    row=original.run(raw_source,training,
                        lambda:data.load_part(p['target_test_ids'],manifest,workspace,clip=False),
                        p,cfg,folder,binding,args.device)
                else:
                    training=clip_part(training);mean,scale=data.label_statistics(clipped_source,training)
                    src=combo.add_z(clipped_source,mean,scale);tr=combo.add_z(training,mean,scale)
                    binding.update(label_mean=float(mean),label_sample_std=float(scale))
                    combo.train_run(src,tr,p,cfg,folder,binding,args.device)
                    testing=data.load_part(p['target_test_ids'],manifest,workspace)
                    row=combo.evaluate(src,testing,p,cfg,folder,binding,mean,scale,args.device)
                    del src,tr,testing
                rows=[r for r in rows if (r['target'],r['train_count'],r['seed'],r['model'])!=(target,count,seed,model)]+[row]
                rows.sort(key=lambda r:order[(r['target'],r['train_count'],r['seed'],r['model'])])
                write_summary(rows,workspace,len(jobs));print(json.dumps(row,ensure_ascii=False),flush=True)
                del training;torch.cuda.empty_cache()
            common.atomic_json(dict(status='complete',completed_runs=len(rows),total_runs=len(jobs)),workspace/'status.json')
            report(workspace)
        except BaseException as error:
            common.atomic_json(dict(status='failed',error=repr(error),traceback=traceback.format_exc()),workspace/'status.json');raise


if __name__=='__main__':main()
