"""Author's two-CNN transfer implementation, with documented correctness repairs."""
import json
import math
import os
import time
from pathlib import Path

import torch
from torch import nn
from scripts import run_three_tasks as common

MODEL='batlinet_author_transfer'


class ConvModule(nn.Module):
    # Same layer names and computations as transfer_reproduce/finalized.ipynb.
    def __init__(self,din,dout,kernel_size,act_fn='relu',dropout=.1):
        super().__init__();self.kernel_size=kernel_size
        self.conv1=nn.Conv2d(din,dout,kernel_size);self.pool1=nn.AvgPool2d(kernel_size)
        self.conv2=nn.Conv2d(dout,dout,kernel_size);self.pool2=nn.AvgPool2d(kernel_size)
        self.act_fn=getattr(torch,act_fn);self.dropout=nn.Dropout2d(dropout)

    def output_shape(self,H,W):
        for _ in range(2):
            H,W=H-self.kernel_size[0]+1,W-self.kernel_size[1]+1
            H=int((H-self.kernel_size[0])/self.kernel_size[0]+1)
            W=int((W-self.kernel_size[1])/self.kernel_size[1]+1)
        return H,W

    def forward(self,x):
        # The author declares dropout but does not apply it in this forward.
        x=self.pool1(self.act_fn(self.conv1(x)))
        return self.pool2(self.act_fn(self.conv2(x)))


class CNNRULPredictor(nn.Module):
    def __init__(self,in_channels,channels,input_height,input_width,kernel_size=3,act_fn='relu'):
        super().__init__();self.channels=channels
        if isinstance(kernel_size,int):kernel_size=(kernel_size,kernel_size)
        kernel_size=(min(input_height,kernel_size[0]),min(input_width,kernel_size[1]))
        self.encoder=ConvModule(in_channels,channels,kernel_size,act_fn)
        H,W=self.encoder.output_shape(input_height,input_width)
        if min(H,W)<1:raise ValueError('卷积输入太小。')
        self.proj=nn.Conv2d(channels,channels,(H,W));self.fc=nn.Linear(channels,1)

    def forward(self,feature):
        x=self.proj(self.encoder(feature)).view(-1,self.channels)
        return self.fc(torch.relu(x)).view(-1)


def make_branches(cfg,seed,device):
    common.seed_all(seed)
    # Do not instantiate or train the separate pretrain/fine-tune baseline.
    return {name:CNNRULPredictor(**cfg['network']).to(device) for name in ('own','relation')}


def clean_difference(x,threshold=10.):
    x=x.clone();x[x.abs()>threshold]=0
    if not torch.isfinite(x).all():raise ValueError('原始模型差分输入非有限。')
    return x


def own_input(feature,cfg):
    base=cfg['preprocessing']['cycle_diff_base']
    return clean_difference(feature-feature[:,:,[base]],cfg['preprocessing']['clip_absolute_over'])


def statistics(source,target,p):
    own_mean=target['label'].mean();own_std=target['label'].std(unbiased=True) if len(target['label'])>1 else torch.tensor(0.)
    fallback=not torch.isfinite(own_std) or own_std<=0
    if fallback:own_mean=source['label'].mean();own_std=source['label'].std(unbiased=True)
    pairs=p['train_pairs'];n=len(source['label'])
    delta=target['label'][pairs//n]-source['label'][pairs%n]
    delta_mean=delta.mean();delta_std=delta.std(unbiased=True)
    if any(not torch.isfinite(v) for v in (own_mean,own_std,delta_mean,delta_std)) or own_std<=0 or delta_std<=0:
        raise ValueError('原始迁移标签统计量不可用。')
    return dict(own_mean=own_mean,own_std=own_std,delta_mean=delta_mean,delta_std=delta_std,
                own_fallback_to_source=bool(fallback),units='cycles_not_log')


def batch(source,target,indices,branch,stats,cfg,device):
    if branch=='own':
        x=own_input(target['feature'][indices].to(device),cfg)
        y=(target['label'][indices].to(device)-stats['own_mean'].to(device))/stats['own_std'].to(device)
    else:
        n=len(source['label']);ti,si=indices//n,indices%n
        # Author clips AFTER subtraction; raw source/target features must be retained.
        x=clean_difference(target['feature'][ti].to(device)-source['feature'][si].to(device),cfg['preprocessing']['clip_absolute_over'])
        y=(target['label'][ti].to(device)-source['label'][si].to(device)-stats['delta_mean'].to(device))/stats['delta_std'].to(device)
    return x,y


def train_epoch(model,source,target,indices,branch,stats,cfg,optimizer,device):
    model.train();logical=cfg['training'][branch+'_batch'];micro=cfg['training']['micro_batch']
    total=0.;steps=0
    for start in range(0,len(indices),logical):
        chosen=indices[start:start+logical];optimizer.zero_grad(set_to_none=True)
        for offset in range(0,len(chosen),micro):
            subset=chosen[offset:offset+micro];x,y=batch(source,target,subset,branch,stats,cfg,device)
            loss=(model(x)-y).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('原始迁移训练损失非有限。')
            (loss*len(subset)/len(chosen)).backward();total+=float(loss.detach())*len(subset)
        if any(t.grad is not None and not torch.isfinite(t.grad).all() for t in model.parameters()):
            raise FloatingPointError('原始迁移梯度非有限。')
        optimizer.step();steps+=1
        if any(not torch.isfinite(t).all() for t in model.parameters()):raise FloatingPointError('原始迁移权重非有限。')
    return dict(train_loss=total/len(indices),optimizer_steps=steps)


@torch.no_grad()
def validate(model,source,target,indices,stats,cfg,device):
    model.eval();total=0.
    for start in range(0,len(indices),cfg['training']['micro_batch']):
        part=indices[start:start+cfg['training']['micro_batch']]
        x,y=batch(source,target,part,'relation',stats,cfg,device)
        value=(model(x)-y).square().sum()
        if not torch.isfinite(value):raise FloatingPointError('原始迁移验证损失非有限。')
        total+=float(value)
    return total/len(indices)


def checked(path,binding):
    state=common.load(path)
    if state['binding']!=binding:raise ValueError('原始迁移权重绑定不同。')
    return state


def train_stage(source,target,p,stats,cfg,folder,binding,branch,device):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    binding=dict(binding,branch=branch)
    maximum=cfg['training'][branch+'_epochs'];patience=cfg['training']['relation_patience']
    final=folder/'final.pt';progress=folder/'progress.pt'
    if final.exists():
        saved=checked(final,binding)
        if not 1<=saved['selected_epoch']<=saved['trained_epochs']<=maximum:raise ValueError('原始迁移完成轮次错误。')
        if branch=='own' and saved['selected_epoch']!=maximum:raise ValueError('自身分支不是固定最终权重。')
        common.align_log(folder,saved['trained_epochs']);return saved
    branches=make_branches(cfg,p['seed'],device);model=branches[branch];del branches
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'])
    generator=torch.Generator().manual_seed(p['seed']);start=0;best=None;best_loss=None;best_epoch=0;bad=0
    if progress.exists():
        checkpoint=checked(progress,binding);start=checkpoint['epoch'];best=checkpoint['best_state']
        best_epoch=checkpoint['best_epoch'];best_loss=checkpoint['best_loss'];bad=checkpoint['bad_epochs']
        if not 0<=start<=maximum or not 0<=bad<=patience:raise ValueError('原始迁移恢复状态错误。')
        model.load_state_dict(checkpoint['state']);optimizer.load_state_dict(checkpoint['optimizer'])
        common.restore_rng(checkpoint['rng'],generator,device)
    common.align_log(folder,start)
    indices=torch.arange(len(target['label'])) if branch=='own' else p['train_pairs']
    def checkpoint(epoch):
        common.save(dict(binding=binding,epoch=epoch,state=model.state_dict(),optimizer=optimizer.state_dict(),
            best_state=best,best_loss=best_loss,best_epoch=best_epoch,bad_epochs=bad,
            rng=common.rng_state(generator,device)),progress)
    if start==0:checkpoint(0)
    epoch=start
    for epoch in range(start+1,maximum+1):
        if branch=='relation' and bad>=patience:epoch-=1;break
        begin=time.monotonic();measured=train_epoch(model,source,target,indices,branch,stats,cfg,optimizer,device)
        val=None
        if branch=='relation':
            val=validate(model,source,target,p['val_pairs'],stats,cfg,device)
            if best_loss is None or val<best_loss:
                best_loss=val;best_epoch=epoch;bad=0
                best={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            else:bad+=1
        else:
            # No own-branch validation or test-based selection in author's transfer routine.
            best_epoch=epoch
        record=dict(epoch=epoch,branch=branch,**measured,validation_relation_mse=val,
            best_epoch=best_epoch,bad_epochs=bad,seconds=time.monotonic()-begin)
        with (folder/'train.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(record,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        checkpoint(epoch)
        stage_label='自身分支' if branch=='own' else '跨电池分支'
        print(f"原始迁移 {p['target_chemistry']} 训练{p['target_train_count']}块 种子{p['seed']} {stage_label} 轮次{epoch} 损失{measured['train_loss']:.6f}",flush=True)
    if branch=='own':best={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    if best is None:raise ValueError('原始迁移未获得有效权重。')
    saved=dict(binding=binding,state=best,selected_epoch=best_epoch,trained_epochs=epoch,
        validation_relation_mse=best_loss,selection='fixed_final_epoch' if branch=='own' else 'first_lowest_validation_relation_mse')
    common.save(saved,final);return saved


def positive_median(values):
    counts=(values>0).sum(1)
    if (counts==0).any():raise FloatingPointError('作者正预测中位数规则遇到全部非正参考预测；保留诊断并停止，不剔除测试电池。')
    return torch.stack([row[row>0].median() for row in values]),counts


@torch.no_grad()
def predict(branches,source,target,stats,cfg,device):
    for model in branches.values():model.eval()
    own=[];relations=[];chunk=cfg['training']['inference_chunk']
    for i in range(len(target['label'])):
        x=own_input(target['feature'][i:i+1].to(device),cfg)
        own.append((branches['own'](x)*stats['own_std'].to(device)+stats['own_mean'].to(device)).cpu().reshape(()))
        row=[]
        for start in range(0,len(source['label']),chunk):
            feature=clean_difference(target['feature'][i:i+1].to(device)-source['feature'][start:start+chunk].to(device),cfg['preprocessing']['clip_absolute_over'])
            delta=branches['relation'](feature)*stats['delta_std'].to(device)+stats['delta_mean'].to(device)
            row.append((delta+source['label'][start:start+chunk].to(device)).cpu())
        relations.append(torch.cat(row))
    y_ori=torch.stack(own);y_sup=torch.stack(relations)
    result=dict(truth_cycles=target['label'],source_ids=source['ids'],target_test_ids=target['ids'],
        statistics=stats,diagnostic_units='cycles',diagnostics=dict(y_ori=y_ori,y_sup=y_sup,
            support_index=torch.arange(len(source['ids'])).expand(len(target['ids']),-1).clone()))
    if not torch.isfinite(y_ori).all() or not torch.isfinite(y_sup).all():raise FloatingPointError('原始迁移预测非有限。')
    if ((y_sup>0).sum(1)==0).any():return dict(result,valid=False,failure='no_positive_reference_prediction')
    agg,counts=positive_median(y_sup);result['diagnostics'].update(y_sup_agg=agg,positive_reference_count=counts)
    prediction=(y_ori+agg)*.5
    result.update(valid=True,prediction=prediction,prediction_cycles=prediction)
    return result


def run(source,target,target_test_loader,p,cfg,folder,binding,device):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    stats=statistics(source,target,p);binding=dict(binding,label_statistics={k:float(v) if torch.is_tensor(v) else v for k,v in stats.items()})
    receipt=folder/'run.json';final=folder/'final.pt';test=folder/'test.pt'
    if receipt.exists() and json.loads(receipt.read_text(encoding='utf-8'))['binding']!=binding:raise ValueError('原始迁移已有收据绑定不同。')
    if not final.exists():
        common.atomic_json(dict(status='training',binding=binding),receipt)
        trained={stage:train_stage(source,target,p,stats,cfg,folder/stage,binding,stage,device) for stage in ('relation','own')}
        saved=dict(binding=binding,state={k:v['state'] for k,v in trained.items()},
            selected_epoch=trained['relation']['selected_epoch'],trained_epochs=trained['relation']['trained_epochs'],
            own_selected_epoch=trained['own']['selected_epoch'],own_trained_epochs=trained['own']['trained_epochs'],
            stages={stage:dict(selected_epoch=value['selected_epoch'],trained_epochs=value['trained_epochs'],
                selection=value['selection'],sha256=common.digest(folder/stage/'final.pt')) for stage,value in trained.items()})
        common.save(saved,final)
    saved=checked(final,binding);weight_sha=common.digest(final)
    if receipt.exists():
        old=json.loads(receipt.read_text(encoding='utf-8'))
        if old.get('final_sha256') is not None and old['final_sha256']!=weight_sha:raise ValueError('原始迁移最佳权重指纹变化。')
        if test.exists() and old.get('test_sha256') is not None and old['test_sha256']!=common.digest(test):raise ValueError('原始迁移测试指纹变化。')
    for stage,metadata in saved['stages'].items():
        if common.digest(folder/stage/'final.pt')!=metadata['sha256']:raise ValueError('原始迁移分支权重指纹变化。')
    testing=target_test_loader()
    if test.exists():output=common.load(test)
    else:
        branches=make_branches(cfg,p['seed'],device)
        for stage,model in branches.items():model.load_state_dict(saved['state'][stage])
        output=predict(branches,source,testing,stats,cfg,device)
        if not output['valid']:
            common.save(dict(output,binding=binding,checkpoint_sha256=weight_sha),folder/'test_failure.pt')
            raise FloatingPointError('原始模型全部参考预测非正，诊断保存于test_failure.pt；不跳过种子。')
        output.update(binding=binding,checkpoint_sha256=weight_sha,selected_epoch=saved['selected_epoch'],
            trained_epochs=saved['trained_epochs'],own_selected_epoch=saved['own_selected_epoch'],
            own_trained_epochs=saved['own_trained_epochs'],scores=common.scores(output['prediction_cycles'],output['truth_cycles']))
        common.save(output,test)
    if (output['binding']!=binding or output['checkpoint_sha256']!=weight_sha or output['diagnostic_units']!='cycles'
            or output['source_ids']!=p['source_ids'] or output['target_test_ids']!=p['target_test_ids']):
        raise ValueError('原始迁移测试绑定、单位或编号不同。')
    for key in ('selected_epoch','trained_epochs','own_selected_epoch','own_trained_epochs'):
        if output[key]!=saved[key]:raise ValueError('原始迁移测试轮次与权重不同。')
    torch.testing.assert_close(output['truth_cycles'],testing['label'],rtol=0,atol=0)
    torch.testing.assert_close(output['diagnostics']['support_index'],torch.arange(len(source['ids'])).expand(len(testing['ids']),-1),rtol=0,atol=0)
    agg,counts=positive_median(output['diagnostics']['y_sup'])
    torch.testing.assert_close(agg,output['diagnostics']['y_sup_agg'])
    torch.testing.assert_close(counts,output['diagnostics']['positive_reference_count'],rtol=0,atol=0)
    torch.testing.assert_close(output['prediction_cycles'],.5*(output['diagnostics']['y_ori']+agg))
    scores=common.scores(output['prediction_cycles'],output['truth_cycles'])
    for key,value in scores.items():
        if not math.isclose(value,output['scores'][key],abs_tol=1e-10,rel_tol=1e-10):raise ValueError('原始迁移测试指标不能复算。')
    common.atomic_json(dict(status='complete',binding=binding,final_sha256=weight_sha,test_sha256=common.digest(test),
        scores=scores,selected_epoch=saved['selected_epoch'],trained_epochs=saved['trained_epochs'],
        own_selected_epoch=saved['own_selected_epoch'],own_trained_epochs=saved['own_trained_epochs']),receipt)
    return dict(target=p['target_chemistry'],train_count=p['target_train_count'],model=MODEL,seed=p['seed'],
        selected_epoch=saved['selected_epoch'],trained_epochs=saved['trained_epochs'],**scores)
