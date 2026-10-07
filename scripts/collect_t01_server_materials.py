"""服务器材料只读打包工具。标准库即可；不导入模型、不训练、不修改源结果。
服务器用法：python -B scripts/collect_t01_server_materials.py --root /root/autodl-tmp
默认读取 /root/autodl-tmp，将新压缩包写到该目录，输出完整下载路径。
"""
import argparse, datetime, hashlib, io, json, subprocess, tarfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/root/autodl-tmp'))
    args=p.parse_args();root=args.root.resolve(); selected={};missing=[];expected=[]
    def add(path,role):
        path=path.resolve()
        if not path.is_relative_to(root):
            missing.append({'path':str(path),'reason':'不在指定材料根目录内'});return
        if path.is_file():selected[path]=role
        else:missing.append({'path':str(path),'reason':'不存在'})
    def folder(path,role):
        if not path.is_dir():missing.append({'path':str(path),'reason':'目录不存在'});return
        for f in sorted(path.rglob('*')):
            if f.is_file() and not f.is_symlink() and '__pycache__' not in f.parts:add(f,role)
    mixer=root/'BatLiNet-cycle-mixer'
    suites=[('mix20_cycle_mixer_207_v1',['latent_cycle_mixer']),
            ('mix20_latent_current_207_v1',['latent_cross_attention_current']),
            ('mix100_cycle_mixer_v1',['latent_cross_attention_current','latent_cycle_mixer'])]
    for suite,models in suites:
        base=mixer/'workspaces'/suite
        add(base/'summary.json','主实验汇总')
        if (base/'protocols').is_dir():folder(base/'protocols','主实验实际固定参考')
        for model in models:
            for seed in range(8):
                run=base/f'{model}_seed{seed}'
                for name in ('run.json','train.jsonl','epoch1000.pt','test.pt'):
                    f=run/name;expected.append(str(f));add(f,'主实验'+suite)
                # MIX-20旧入口引用外部协议，按运行记录收集实际路径。
                meta=run/'run.json'
                if meta.exists():
                    try:
                        info=json.loads(meta.read_text(encoding='utf-8'))
                        prov=info.get('provenance',{})
                        for path in prov.get('protocols_paths',[]):add(Path(path),'运行记录实际固定参考')
                    except (ValueError,OSError) as e:missing.append({'path':str(meta),'reason':str(e)})
    # 三模型范围内的两个补充批次，严格选择指定模型子目录。
    for seed in range(8):
        folder(mixer/'workspaces/mix20_cycle_mixer_v1'/f'latent_cycle_mixer_seed{seed}','166／41循环轴训练材料')
        add(mixer/'evaluations/mix20_cycle_mixer_v1/latent_cycle_mixer'/f'seed{seed}.pt','166／41循环轴正式预测')
        raw=mixer/'workspaces/mix20_cycle_difference_207_v1'/f'cycle_raw_current_seed{seed}'
        for name in ('run.json','train.jsonl','epoch1000.pt','test.pt'):add(raw/name,'另次原始输入循环轴批次，单独标注')
        old=root/'BatLiNet-context-grid-v1'
        for model in ('batlinet','latent_cross_attention'):
            add(old/'workspaces/mix20_matched_baselines_v1'/f'{model}_seed{seed}'/'best.pt','166／41两个旧模型正式权重')
        for part in ('test','val'):add(old/'protocols/mix20_matched_v1'/f'{part}_seed{seed}.pt','166／41实际参考')
    add(root/'BatLiNet-context-grid-v1/artifacts/mix20_matched_baselines_v1/mix20.pt','166／41三模型共有原始六通道数据缓存')
    add(root/'BatLiNet-context-grid-v1/artifacts/mix20_matched_baselines_v1/mix20.json','166／41数据元信息')
    for repo in ('BatLiNet-main2','BatLiNet-main3'):
        for task in ('mix_20','mix_100'):
            for seed in range(8):
                f=root/repo/'artifacts/fixed_test_support_indices'/task/'protocol_v1'/f'seed_{seed}.pt'
                if f.is_file():add(f,'历史独立固定参考')
    # 当前相关代码用于追溯；当前快照不冒充训练时版本。
    for rel in ('scripts/collect_t01_server_materials.py',
                'src/models/rul_predictors/batlinet.py','src/models/rul_predictors/latent_cross_attention_batlinet.py',
                'src/models/rul_predictors/cycle_mixer_latent_cross_attention_batlinet.py',
                'src/models/rul_predictors/cycle_difference_batlinet.py','src/models/nn_model.py',
                'src/data/databundle.py','src/data/transformation/z_score.py','src/data/transformation/log_scale.py',
                'src/data/transformation/sequential.py','src/feature/batlinet.py','src/label/rul.py',
                'scripts/run_mix20_cycle_mixer_207.py','scripts/run_mix20_latent_current_207.py',
                'scripts/run_mix100_cycle_mixer.py','scripts/run_mix20_cycle_difference_207.py',
                'scripts/matched_baselines.py','scripts/evaluate_cycle_mixer_suite.py','scripts/pipeline.py',
                'configs/ablation/diff_branch/batlinet_latent_cross_attention/mix_20.yaml',
                'configs/ablation/diff_branch/batlinet_latent_cross_attention/mix_100.yaml'):
        add(mixer/rel,'当前相关源码快照；仅用于版本核对')
    # 只收集三项主实验日志；不打包无关实验日志。
    for f in sorted((mixer/'logs').glob('*')):
        if f.is_file() and any(s in f.name for s in ('mix100_cycle_mixer','mix20_cycle_mixer_207','mix20_latent_current_207')):
            add(f,'主实验服务器完整日志')
    manifest=[]
    for f,role in sorted(selected.items(),key=lambda x:str(x[0])):
        h=hashlib.sha256()
        with f.open('rb') as stream:
            for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
        manifest.append({'path':str(f.relative_to(root)),'bytes':f.stat().st_size,'sha256':h.hexdigest(),'role':role})
    meta={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),
          'files':manifest,'missing':missing,'required_main_files':expected,
          'scope':'只读取三模型既有结果，不训练，不运行评价，不修改源文件'}
    try:
        r=subprocess.run(['git','-C',str(mixer),'rev-parse','HEAD'],capture_output=True,text=True)
        meta['current_head']=r.stdout.strip()
        r=subprocess.run(['git','-C',str(mixer),'--no-optional-locks','status','--short','--untracked-files=no'],capture_output=True,text=True)
        meta['current_tracked_changes']=r.stdout.strip()
    except OSError:meta['git']='不可用'
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output=root/f'T01_三模型材料_{stamp}.tar.gz'
    with output.open('xb') as target,tarfile.open(fileobj=target,mode='w:gz') as tar:
        for f in sorted(selected):tar.add(f,arcname=str(f.relative_to(root)),recursive=False)
        b=json.dumps(meta,ensure_ascii=False,indent=2).encode('utf-8')
        item=tarfile.TarInfo('T01_服务器材料清单.json');item.size=len(b);tar.addfile(item,io.BytesIO(b))
    print('打包完成：',output)
    print('文件数：',len(selected),'；缺失条目：',len(missing),'；压缩包字节：',output.stat().st_size)
    print('请下载这个压缩包。缺失清单已写入包内，不会补训练或覆盖结果。')

if __name__=='__main__':main()
