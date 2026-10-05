# 潜在自注意力与交叉注意力模型：版本 A

本文件记录本次实现和运行方式，不包含尚未运行的性能结论。

## 实现范围

注册类：`LatentSelfCrossAttentionBatLiNetRULPredictor`。

在原 `ConvTokenEncoder` 的二维卷积、池化、展开及可学习位置参数之后，新增一层预归一化自注意力与残差前馈网络。目标和参考各自独立编码，使用同一组参数，不在电池之间执行这层自注意力。

默认隐藏维度 64、4 个注意力头、前馈维度 128、随机失活率 0.1；新增参数 33,472。MIX-20 输出仍为 155×64，MIX-100 仍为 775×64。窗口内输入均已观测，自注意力不使用因果掩码。

本次没有增加原始／变化双路输入。输入提取、清理、目标—参考交叉注意力、两个回归头、标签变换、分支损失、训练参考数 2、测试参考数 32、训练均值／测试较小中位值聚合、alpha=0.5 均继承基础模型。设置 `self_attention_layers: 0` 时不包装编码器，模型权重键和预测路径与基础模型兼容。

本模型没有新增有效位置掩码；MIX-20 索引 10 的循环仍按原配置置零。卷积和位置参数作用后的局部向量不能简单作为全零循环直接屏蔽。

## 文件

- `src/models/rul_predictors/latent_self_cross_attention_batlinet.py`：模型。
- `configs/ablation/diff_branch/batlinet_latent_self_cross_attention/mix_20.yaml`。
- `configs/ablation/diff_branch/batlinet_latent_self_cross_attention/mix_100.yaml`。
- `scripts/test_latent_self_cross_attention.py`：无真实电池数据的 CPU 检查。
- `scripts/run_latent_self_cross_attention.sh`：单设备顺序训练与固定协议评估。
- `src/utils/training_progress.py`：逐行训练进度及剩余时间估算。
- `scripts/summarize_latent_self_cross_attention.py`：预测指标重算与逐种子汇总。
- `scripts/test_training_progress_and_results.py`：日志、优化一致性和结果汇总检查。

## 实验设置

采用基础潜在交叉注意力模型的历史 MIX 划分和配置，不使用 v1 的 166／41 训练验证划分。训练 1000 轮，保存并评价第 1000 轮权重，不按测试误差选权重或种子。训练循环继承每 100 轮测试误差输出，这不是独立验证集，不能称为验证选权重。

新配置显式设置 `checkpoint_freq: 1000`。原配置中的其他实验字段保持一致，仅模型注册名、自注意力字段和这一保存频率字段不同。周期测试仍按基础模型抽取随机参考，正式评估才加载历史固定参考名单。训练脚本不会重新生成参考协议。

## 本地检查

在仓库根目录、已有 batlinet 环境运行：

```bash
python -B scripts/test_latent_self_cross_attention.py
python -B scripts/test_training_progress_and_results.py
```

检查覆盖配置与注册、20／100 循环前向及反向、两个分支和新增模块梯度、均值／较小中位值聚合、关闭自注意力时与基础模型一致、独立电池编码、固定 32 参考、参考重排、权重保存加载及一次完整训练保存。

## 服务器运行

本地开发分支：`agent/latent-self-cross-attention`。服务器创建独立工作树，不切换已有训练仓库：

```bash
cd /root/autodl-tmp/BatLiNet-main3
git fetch origin agent/latent-self-cross-attention
git worktree add --detach /root/autodl-tmp/BatLiNet-latent-self-attention origin/agent/latent-self-cross-attention
cd /root/autodl-tmp/BatLiNet-latent-self-attention
conda activate batlinet
mkdir -p data
ln -s /root/autodl-tmp/BatLiNet-main2/data/processed data/processed
python -B scripts/test_latent_self_cross_attention.py
```

工作树已存在时，先核对路径和本地改动，不重复执行 `git worktree add`。数据链接已经存在时不重复执行 `ln -s`。

使用 `nohup` 运行 MIX-20 的八个种子，日志追加写入，标准输入与终端断开：

```bash
mkdir -p logs
nohup bash scripts/run_latent_self_cross_attention.sh mix_20 all \
  >> logs/latent_self_cross_attention_mix20.nohup.log 2>&1 < /dev/null &
echo "$!" > logs/latent_self_cross_attention_mix20.pid
```

MIX-100：

```bash
nohup bash scripts/run_latent_self_cross_attention.sh mix_100 all \
  >> logs/latent_self_cross_attention_mix100.nohup.log 2>&1 < /dev/null &
echo "$!" > logs/latent_self_cross_attention_mix100.pid
```

以上两项在单张 GPU 上应顺序运行。`nohup` 保持 SSH 断线后任务运行；它不提供服务器重启后的恢复或训练断点续训。

Python 使用无缓冲输出。日志包含模型、任务、种子、本次第几次训练、轮次、本轮批次、阶段、已耗时、当前训练预计剩余时间，以及本次种子范围内后续待训练次数。每轮结束输出一次，轮内在批次结束时检查间隔（默认 30 秒，`PROGRESS_INTERVAL_SECONDS` 可覆盖）。首次尚无完成批次时显示“估算中”；估计采用当前种子的已耗时／完成批次数，周期测试耗时也进入估计，首次编译及不同阶段会使估计波动。时间显示为北京时间。

日志钩子不改变训练计算或随机抽样；基础模型未开启钩子时继续原进度显示。固定随机种子的两轮训练及周期测试，开启日志和关闭日志所得权重逐项一致的 CPU 检查通过。

单独查看实时日志，按 Ctrl+C 只退出查看：

```bash
tail -n 60 -f logs/latent_self_cross_attention_mix20.nohup.log
```

单独查看最终结果：

```bash
python -B scripts/summarize_latent_self_cross_attention.py mix_20
```

MIX-100 将日志文件名中的 `mix20` 改为 `mix100`、结果命令的任务改为 `mix_100`。结果命令显示各个种子的 RMSE、MAE、MAPE，以及全部预期种子的均值与样本标准差。MAPE 在终端显示百分数，JSON 保留比例值。种子缺失时只显示已完成指标和缺失列表，不生成最终汇总；重复预测文件或重算指标与保存指标不一致时报错。

运行脚本完成全部正式评估后，会自动保存 `summary_seed_0_7.json`（若种子范围不同，文件名相应改变），并在总日志输出汇总和 `[全部完成]`。单独运行结果查看命令默认只读取文件。自定义 `EVAL_WS` 时，查看结果需同时传入 `--workspace` 对应路径。

默认协议分别位于：

```text
/root/autodl-tmp/BatLiNet-main2/artifacts/fixed_test_support_indices/mix_20/protocol_v1
/root/autodl-tmp/BatLiNet-main3/artifacts/fixed_test_support_indices/mix_100/protocol_v1
```

结果分别保存至：

```text
workspaces/fair_train/batlinet_latent_self_cross_attention/<任务>
workspaces/fair_eval/batlinet_latent_self_cross_attention/<任务>/protocol_v1_formal
```

评估目录保存预测、分支诊断、实际参考索引、数据元信息、配置快照和日志。训练目录保存各种子的第 1000 轮权重、配置快照和日志。

脚本支持 `train`、`evaluate`、`all` 三种模式，已有权重／预测时拒绝覆盖；使用 `START_SEED` 和 `END_SEED` 指定待运行范围，例如：

```bash
START_SEED=3 END_SEED=7 bash scripts/run_latent_self_cross_attention.sh mix_20 all
```

这会从头训练种子 3～7，不是从某个训练轮次续训。若某种子已经训练完成但没有正式预测，先单独评估该种子，再训练其他种子：

```bash
START_SEED=2 END_SEED=2 bash scripts/run_latent_self_cross_attention.sh mix_20 evaluate
```

可用 `DEVICE`、`PROTOCOL_DIR`、`TRAIN_WS`、`EVAL_WS` 覆盖默认路径／设备。同一任务必须继续使用历史协议及相同数据顺序；不得直接使用 v1 的 166 电池参考索引。MIX-100 的实际 GPU 显存占用尚未测量。
