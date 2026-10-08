"""Paper-defined source/target split and author-derived early feature preparation."""
import gc
import hashlib
import json
import math
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from scripts import run_three_tasks as common
from src.data import BatteryData
from src.feature import BatLiNetFeatureExtractor
from src.label import RULLabelAnnotator

SOURCES = ('MATR', 'HUST', 'CALCE', 'HNEI', 'RWTH', 'SNL', 'UL_PUR')
EXPECTED = {'LFP': 275, 'LCO': 37, 'NCA': 22, 'NMC': 69}
TEST_COUNTS = {'LCO': 21, 'NCA': 14, 'NMC': 53}
TRAIN_COUNTS = {'LCO': (1, 2, 4, 8, 16), 'NCA': (1, 2, 4, 8), 'NMC': (1, 2, 4, 8, 16)}
POLICY = dict(version=1, source_chemistry='LFP', source_count=275,
    target_total_counts={k:v for k,v in EXPECTED.items() if k!='LFP'},
    target_test_counts=TEST_COUNTS, seeds=list(range(8)),
    split='fixed_target_test_within_seed_across_counts', nested_training_subsets=True,
    pair_validation_fraction=.2, pair_split='numpy_RandomState_seed_permutation',
    feature_shape=[6,100,1000], early_cycles=100, capacity_axis_max=1.2,
    eol_soh=.8, min_rul_limit=100., pad_eol=True, smooth_features=False,
    feature_clip='absolute_value_over_10_to_zero_before_encoder',
    label_space='natural_log_common_zscore', label_statistics='source_and_selected_target_training_only_sample_std',
    evaluation_references='all_275_LFP_cells_in_saved_order',
    selection='first_lowest_validation_relation_mse', test_during_training=False)


def chem(value):
    return str(value).strip().upper()


def feature_path(folder, cell_id):
    return folder/'cells'/(hashlib.sha256(cell_id.encode()).hexdigest()+'.pt')


def validate_counts(records):
    counts=dict(Counter(r['chemistry'] for r in records))
    if counts != EXPECTED:
        raise ValueError(f'有效电池数量与论文及作者缓存记录不同：实际{counts}，预期{EXPECTED}。请检查数据收据；不自动改变协议。')
    if len({r['cell_id'] for r in records}) != len(records):
        raise ValueError('电池ID重复。')
    for r in records:
        if not math.isfinite(r['label_cycles']) or r['label_cycles']<=100:
            raise ValueError('有效寿命必须大于100循环。')
    return counts


def save_cell(battery, material, origin, folder):
    x=torch.as_tensor(battery.feature).detach().cpu().float().contiguous()
    y=float(torch.as_tensor(battery.label).item())
    if tuple(x.shape)!=(6,100,1000) or not torch.isfinite(x).all() or not math.isfinite(y) or y<=100:
        raise ValueError(f'电池输入/寿命不符合作者特征协议：{battery.cell_id}，shape={tuple(x.shape)}，label={y}')
    cell_id=str(battery.cell_id)
    path=feature_path(folder,cell_id)
    value=dict(cell_id=cell_id,chemistry=material,feature=x,label_cycles=y,
               nominal_capacity_in_Ah=float(battery.nominal_capacity_in_Ah))
    # Interrupted preparation can be resumed without overwriting an inconsistent cell.
    if path.exists():
        old=common.load(path)
        if old['cell_id']!=cell_id or old['chemistry']!=material or old['label_cycles']!=y or not torch.equal(old['feature'],x):
            raise ValueError(f'已有特征缓存与源数据不一致：{path}')
    else:
        common.save(value,path)
    return dict(cell_id=cell_id,chemistry=material,label_cycles=y,
        nominal_capacity_in_Ah=value['nominal_capacity_in_Ah'],
        feature_file=path.relative_to(folder).as_posix(),feature_sha256=common.digest(path),
        origin=origin)


def check_manifest(folder, manifest, identity):
    if manifest['identity'] != identity:
        raise ValueError('已有数据收据与来源、处理代码或原数据指纹不同；需独立工作区。')
    validate_counts(manifest['records'])
    for r in manifest['records']:
        p=folder/r['feature_file']
        if not p.is_file() or common.digest(p)!=r['feature_sha256']:
            raise ValueError(f'数据特征缺失或指纹变化：{p}')
    return manifest


def prepare(source_root, workspace, code_hashes, policy=None):
    source_root=Path(source_root).resolve();folder=Path(workspace)/'dataset'
    folder.mkdir(parents=True,exist_ok=True)
    cached=source_root/'cache/transfer.pkl'
    if cached.is_file():
        route=dict(kind='author_transfer_cache',path=str(cached),sha256=common.digest(cached))
        files=[]
    else:
        processed=source_root/'data/processed'
        files=[]
        for name in SOURCES:
            current=sorted((processed/name).glob('*.pkl'))
            if not current:
                raise FileNotFoundError(f'缺少作者迁移所需数据源：{processed/name}；服务器需包含SNL等七个数据源，不能用MIX子集代替。')
            files.extend(current)
        route=dict(kind='processed_cells',root=str(processed),
                   files=[dict(path=str(p),sha256=common.digest(p)) for p in files])
    identity=dict(policy=POLICY if policy is None else policy,source=route,code_hashes=code_hashes)
    manifest_path=folder/'manifest.json'
    if manifest_path.exists():
        return check_manifest(folder,json.loads(manifest_path.read_text(encoding='utf-8')),identity)
    records=[];excluded=[]
    seen=set()
    if cached.is_file():
        # This is the user's own trusted author-format cache; not an untrusted download.
        with cached.open('rb') as f: groups=pickle.load(f)
        for material,batteries in groups.items():
            material=chem(material)
            for battery in batteries:
                if str(battery.cell_id) in seen:raise ValueError('源缓存重复电池ID。')
                seen.add(str(battery.cell_id))
                if material not in EXPECTED:
                    excluded.append(dict(cell_id=str(battery.cell_id),reason='不属于四个预定体系'));continue
                records.append(save_cell(battery,material,dict(cache_sha256=route['sha256']),folder))
                if len(records)%25==0:print(f'已整理作者特征缓存：{len(records)}块电池',flush=True)
        del groups;gc.collect()
    else:
        labeler=RULLabelAnnotator(eol_soh=.8,pad_eol=True,min_rul_limit=100)
        extractor=BatLiNetFeatureExtractor(smooth_features=False,interp_dim=1000,
            min_cycle_index=0,max_cycle_index=99,max_capacity=1.2)
        for index,path in enumerate(files):
            battery=BatteryData.load(path);material=chem(battery.cathode_material)
            if str(battery.cell_id) in seen:raise ValueError('源数据重复电池ID。')
            seen.add(str(battery.cell_id))
            if material not in EXPECTED:
                excluded.append(dict(path=str(path),cell_id=str(battery.cell_id),reason='不属于四个预定体系'));continue
            battery.label=labeler.process_cell(battery)
            if not torch.isfinite(battery.label):
                excluded.append(dict(path=str(path),cell_id=str(battery.cell_id),reason='作者寿命过滤'));continue
            battery.feature=extractor.process_cell(battery)
            records.append(save_cell(battery,material,dict(path=str(path),sha256=route['files'][index]['sha256']),folder))
            del battery
            if (index+1)%10==0:print(f'迁移特征整理：{index+1}/{len(files)}，有效{len(records)}',flush=True)
    records.sort(key=lambda r:r['cell_id'])
    audit=dict(identity=identity,records=records,excluded=excluded,
               observed_counts=dict(Counter(r['chemistry'] for r in records)),
               feature_and_label_independently_rebuilt=not cached.is_file())
    common.atomic_json(audit,folder/'preparation_receipt.json')
    validate_counts(records)
    common.atomic_json(audit,manifest_path)
    return audit


def build_protocol(records, target, count, seed, policy=None):
    validate_counts(records)
    if target not in TRAIN_COUNTS or count not in TRAIN_COUNTS[target] or seed not in range(8):
        raise ValueError('化学体系、训练数量或种子不在冻结协议中。')
    source=sorted(r['cell_id'] for r in records if r['chemistry']=='LFP')
    target_ids=sorted(r['cell_id'] for r in records if r['chemistry']==target)
    tests=random.Random(seed).sample(target_ids,TEST_COUNTS[target])
    candidates=sorted(set(target_ids)-set(tests))
    random.Random(1000+seed).shuffle(candidates)
    training=candidates[:count]
    n_pairs=count*len(source)
    order=np.random.RandomState(seed).permutation(n_pairs)
    cut=int(n_pairs*.8)
    train_pairs=torch.from_numpy(order[:cut].copy()).long()
    val_pairs=torch.from_numpy(order[cut:].copy()).long()
    if not len(train_pairs) or not len(val_pairs):raise ValueError('电池对划分为空。')
    return dict(policy=POLICY if policy is None else policy,target_chemistry=target,target_train_count=count,seed=seed,
                source_ids=source,target_train_ids=training,target_test_ids=tests,
                unused_target_ids=candidates[count:],train_pairs=train_pairs,val_pairs=val_pairs)


def validate_protocol(p, records, policy=None):
    expected=build_protocol(records,p['target_chemistry'],p['target_train_count'],p['seed'],policy)
    for key,value in expected.items():
        if torch.is_tensor(value):
            if not torch.equal(p[key],value):raise ValueError(f'协议数组被修改：{key}')
        elif p[key]!=value:raise ValueError(f'协议字段被修改：{key}')
    source=set(p['source_ids']);train=set(p['target_train_ids']);test=set(p['target_test_ids'])
    if source&train or source&test or train&test:raise ValueError('训练/参考/测试电池重叠。')
    pairs=torch.cat((p['train_pairs'],p['val_pairs']))
    if sorted(pairs.tolist())!=list(range(len(source)*len(train))):raise ValueError('电池对划分重复或缺失。')


def protocol(manifest, target, count, seed, workspace, policy=None):
    path=Path(workspace)/'protocols'/target/f'train_{count}'/f'seed_{seed}.pt'
    p=build_protocol(manifest['records'],target,count,seed,policy)
    if path.exists():p=common.load(path)
    else:common.save(p,path)
    validate_protocol(p,manifest['records'],policy)
    return p,path


def load_part(ids, manifest, workspace, clip=True):
    lookup={r['cell_id']:r for r in manifest['records']}
    xs=[];ys=[]
    for cell_id in ids:
        record=lookup[cell_id];path=Path(workspace)/'dataset'/record['feature_file']
        if common.digest(path)!=record['feature_sha256']:raise ValueError('加载时特征指纹与收据不同。')
        value=common.load(path)
        if value['cell_id']!=cell_id or value['label_cycles']!=record['label_cycles'] or tuple(value['feature'].shape)!=(6,100,1000):
            raise ValueError('缓存内容与已核验收据不同。')
        x=value['feature'].float().clone()
        if clip:x[x.abs()>10]=0
        if not torch.isfinite(x).all():raise ValueError('模型输入非有限。')
        xs.append(x);ys.append(float(value['label_cycles']))
    return dict(feature=torch.stack(xs),label=torch.tensor(ys),ids=list(ids))


def label_statistics(source, target):
    logs=torch.cat((source['label'],target['label'])).double().log()
    mean,scale=logs.mean().float(),logs.std(unbiased=True).float()
    if not torch.isfinite(scale) or scale<=0:raise ValueError('训练标签统计量不可用。')
    return mean,scale