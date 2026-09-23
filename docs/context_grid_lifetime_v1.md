# 二维局部上下文寿命模型：首版实现与运行说明

本说明对应 `agent/context-grid-lifetime-v1` 的代码，不是已完成的性能报告。
没有改动原模型、既有训练入口或交接文档。此模型使用独立入口，不使用 `scripts/pipeline.py`。

## 实现范围

- 输入为原始电压、倍率、额定容量归一化的容量，以及独立的位置掩码。
- 充放电分别在容量范围 `[0, 1.2]` 上插值，各 256 点。超出实测范围的位置屏蔽；不能将此坐标称为实际 SOC。
- 描述量为充放电容量/额定容量、充放电有效时长、库仑效率、能量效率、额定容量、原始循环编号；缺失量具有独立掩码。
- 只有观测窗口内的循环进入输入；MIX-20 沿用索引 10 循环屏蔽。
- 以最早三个有效循环的逐点中位数构造差分，只在共同有效位置计算。
- 原始量、差分量、描述量的尺度统计仅从实际训练电池拟合，作为模型缓冲区写入检查点。
- 两条无样本级归一化的局部二维卷积通路；充放电分别卷积。
- 独立整循环映射及一层循环注意力；固定容量区间的两层有序时序卷积及首末/均值/最大值汇总。
- 两个上下文通过初始值 0.1 的可训练残差系数加入二维网格，再压缩循环轴。64 维、32 个容量区间；20/100 循环产生160/800 个词元。
- 目标查询参考的单向交叉注意力、目标/参考预测头、各 0.5 的分支损失及融合。
- 训练2参考随机有放回抽样（沿用基础版，可能抽到自身）；测试32参考取 PyTorch 的下中位数。所有计算先在标准化对数寿命空间进行，最后还原循环数。
- `--no-context` 关闭两个上下文模块，保留相同数据、局部卷积、差分和参考框架；它是消融对照，不等同于原始潜在交叉注意力模型。
- 使用训练集内部验证，不在训练过程中评估测试集。CUDA 可选 bfloat16；参考编码在评估时缓存，注意力分块以控制显存。

## 环境与本地验证

沿用仓库 `batlinet` 环境（PyTorch 2.x、NumPy、SciPy 等原有依赖）。RTX 4090 可以使用 `--amp`。
本地 CPU 测试包含早期窗口、充放电与缺失掩码、训练统计与检查点恢复、20/100循环前反向、参考换序、逐对计算、分块评估和小型完整训练评估。

```bash
conda activate batlinet
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
python -B scripts/test_context_grid_lifetime.py
```

CPU 合成数据检查通过不等于真实数据训练有效；没有声称获得新的正式成绩。

## 服务器：先运行 MIX-20 小规模验证

在服务器已有仓库旁建立独立工作目录，避免切换或覆盖旧实验工作目录：

```bash
cd /root/autodl-tmp/BatLiNet-main3
git fetch origin agent/context-grid-lifetime-v1
git worktree add --detach /root/autodl-tmp/BatLiNet-context-grid-v1 origin/agent/context-grid-lifetime-v1
cd /root/autodl-tmp/BatLiNet-context-grid-v1
conda activate batlinet
bash scripts/run_context_grid_pilot.sh /root/autodl-tmp/BatLiNet-main2/data/processed
```

该脚本执行测试、数据准备、20轮训练与验证，不评估测试集。每轮输出损失、训练耗时、CUDA峰值显存；每5轮输出验证指标。工作目录为 `workspaces/context_grid_v1/mix20_pilot_seed0`。
脚本拒绝覆盖已有运行目录；重复试验请传第二个参数指定新目录。首次 worktree 创建命令只需执行一次。

## 正式训练候选与对照

```bash
python scripts/context_grid_lifetime.py train \
  --data artifacts/context_grid_v1/mix20.pt \
  --workspace workspaces/context_grid_v1/mix20_seed0 \
  --seed 0 --epochs 1000 --batch-size 8 --accumulation 16 \
  --evaluate-every 50 --selection-metric RMSE --amp

python scripts/context_grid_lifetime.py train \
  --data artifacts/context_grid_v1/mix20.pt \
  --workspace workspaces/context_grid_v1/mix20_local_only_seed0 \
  --seed 0 --epochs 1000 --batch-size 8 --accumulation 16 \
  --evaluate-every 50 --selection-metric RMSE --amp --no-context
```

使用同一数据文件和种子时，验证参考相同；训练的随机数消耗会随结构改变，不保证训练配对序列逐项一致。
`best.pt` 仅按验证指标选择，`run.json` 保存所有参数、数据SHA256、源数据文件哈希、训练电池清单和代码提交，`train.jsonl` 保存每轮记录。
默认有效批量约128，最后不足批量按实际样本数加权。CPU 模式不要传 `--amp`。
1000轮是候选上限，不是已确认最优轮数。当前不支持断点续训，工作目录拒绝覆盖。

MIX-100 准备：

```bash
python scripts/context_grid_lifetime.py prepare --dataset mix100 --cycles 100 \
  --data-root /root/autodl-tmp/BatLiNet-main2/data/processed \
  --output artifacts/context_grid_v1/mix100.pt
```

训练命令替换数据和工作目录即可。先测显存；必要时将批量8降为4，累积16升为32。降低评估显存可加 `--encode-batch 4 --pair-chunk 4`。

## BatteryLife：明确快照与固定窗口任务

当前下载器默认快照为 `154a4026cd188960b9bfc029a9e28ad3f1091910`，来自2026-09-24查到的官方数据仓库主线。
它不是声称复现论文最初数据或 v11 的标记。新模型与比较模型必须使用同一快照。

```bash
git clone https://github.com/Ruifeng-Tan/BatteryLife.git /root/autodl-tmp/BatteryLife-official
python -m pip install huggingface_hub
python scripts/download_context_grid_batterylife.py \
  --output /root/autodl-tmp/BatteryLife-context-grid-snapshot

python scripts/context_grid_lifetime.py prepare --dataset batterylife --cycles 20 \
  --official-repo /root/autodl-tmp/BatteryLife-official \
  --data-root /root/autodl-tmp/BatteryLife-context-grid-snapshot \
  --data-version 154a4026cd188960b9bfc029a9e28ad3f1091910 \
  --output artifacts/context_grid_v1/batterylife20.pt

python scripts/context_grid_lifetime.py train \
  --data artifacts/context_grid_v1/batterylife20.pt \
  --workspace workspaces/context_grid_v1/batterylife20_seed0 \
  --seed 0 --epochs 1000 --batch-size 8 --accumulation 16 \
  --evaluate-every 50 --selection-metric MAPE --amp
```

如下载返回401/403，先在服务器执行 `hf auth login`，并确认账户具备该数据仓库访问权限。不要把令牌写进代码或命令记录。
本地环境能查询快照文件清单，但获取数据文件返回401，未完成真实 BatteryLife 数据全量验证。

准备器解析官方 `MIX_large_train/val/test_files`，直接使用 `Life labels` 的寿命标签，沿用无标签/EOL<=100过滤及官方额定容量修正。
MICH 原始目录可以使用 `MICH`/`MICH_EXP`，也支持作者预先合并的 `total_MICH`。输入处理采用本模型实现，不声称逐行复现官方数据处理。
记录官方仓库实际提交、划分文件哈希、每个数据/标签文件哈希及排除电池。数据缺失、形状异常或集合交叉会报错，不能静默改划分。
该入口每块电池固定一个20或100循环窗口，不复现官方1–100窗口汇总；因此需要重跑对照，不能直接与官方论文汇总数字比较。
更改为100循环时同时修改 `--cycles` 和输出文件名。不要跨 BatteryLife/MIX 合并训练后直接评价重叠电池。

## 最终评估与旧固定参考协议

开发完成后可以直接对验证选出的模型进行一次测试评估：

```bash
python scripts/context_grid_lifetime.py evaluate \
  --data artifacts/context_grid_v1/mix20.pt \
  --checkpoint workspaces/context_grid_v1/mix20_seed0/best.pt \
  --output workspaces/context_grid_v1/mix20_seed0/test.pt
```

该模型只使用划出的训练子集，不能当作与旧全训练集结果完全匹配的实验。
如需与旧 MIX 正式结果比较，先根据训练内部验证确定固定轮数，再使用 `train --refit --epochs <已确定轮数>` 在训练+验证电池上重训，得到 `final.pt`。
所有候选在相同全训练集、轮数确定规则和固定参考协议下比较。

从旧正式预测文件迁移参考协议（按电池ID重映射，核对训练/测试标签，避免文件遍历顺序造成错配）：

```bash
python scripts/context_grid_lifetime.py protocol \
  --data artifacts/context_grid_v1/mix20.pt --refit --seed 0 \
  --legacy-predictions /完整路径/predictions_seed_0_时间戳.pkl \
  --output artifacts/context_grid_v1/mix20_protocol_seed0.pt

python scripts/context_grid_lifetime.py evaluate \
  --data artifacts/context_grid_v1/mix20.pt \
  --checkpoint /完整路径/全训练集重训目录/final.pt \
  --protocol artifacts/context_grid_v1/mix20_protocol_seed0.pt \
  --output /完整路径/全训练集重训目录/test.pt
```

不能将旧 `.pt` 索引直接套用到新训练顺序。若不指定旧预测，`protocol` 会产生与电池清单绑定的新固定32参考。
评价输出包含预测、真实标签、RMSE/MAE/MAPE/15%准确率、两分支指标、每个参考预测、参考索引、电池清单及数据/检查点指纹。诊断量已还原为寿命循环数。
所有数据、检查点与运行结果均不提交到 Git。
