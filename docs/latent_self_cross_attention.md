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

## 实验设置

采用基础潜在交叉注意力模型的历史 MIX 划分和配置，不使用 v1 的 166／41 训练验证划分。训练 1000 轮，保存并评价第 1000 轮权重，不按测试误差选权重或种子。训练循环继承每 100 轮测试误差输出，这不是独立验证集，不能称为验证选权重。

新配置显式设置 `checkpoint_freq: 1000`。原配置中的其他实验字段保持一致，仅模型注册名、自注意力字段和这一保存频率字段不同。周期测试仍按基础模型抽取随机参考，正式评估才加载历史固定参考名单。训练脚本不会重新生成参考协议。

## 本地检查

在仓库根目录、已有 batlinet 环境运行：

```bash
python -B scripts/test_latent_self_cross_attention.py
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

在 `screen` 会话中运行 MIX-20 的八个种子：

```bash
bash scripts/run_latent_self_cross_attention.sh mix_20 all
```

MIX-100：

```bash
bash scripts/run_latent_self_cross_attention.sh mix_100 all
```

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
