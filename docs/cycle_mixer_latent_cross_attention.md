# 循环轴残差编码器实验

## 实现范围

注册模型名为 `CycleMixerLatentCrossAttentionBatLiNetRULPredictor`。
代码在 `src/models/rul_predictors/cycle_mixer_latent_cross_attention_batlinet.py`。

保留 `latent_cross_attention` 的六通道输入、输入清理、三层二维卷积、位置编码、目标与参考共享参数、交叉注意力、两个预测头、参考抽样和标签变换。
唯一的结构插入点在第三层卷积及激活之后、最后一次二维平均池化之前。
原模型源码和交接文档没有修改。

MIX-20 的中间网格是 `[批量, 64, 10, 62]`，10 行是已经卷积和池化过的循环方向特征，并非十个独立原始循环。
在每个容量位置和特征通道，前馈网络按固定顺序处理这十行，输出十行修正量，残差相加后继续原有池化。
最终仍为 155 个 64 维局部特征。没有新增原始曲线循环差分、描述量或上下文摘要广播。

前馈网络为 `10 → 16 → 10`，激活函数为 GELU，参数在容量位置、特征通道、目标和参考之间共享。
输出层权重和偏置初始化为零。初始化时整个预测器与同种子原模型一致；训练后不保证预测保持一致或误差下降。
辅助模块初始化隔离了随机数状态，也没有新增随机丢弃层，避免额外初始化改变原模型初始权重、训练批次和参考抽样的随机数起点。

## 三组比较

| 训练入口名称 | 编码器 | 新增参数（MIX-20） |
| --- | --- | ---: |
| `latent_cross_attention` | 原共享二维卷积编码器 | 0 |
| `latent_cycle_mixer` | 原编码器加循环轴残差前馈网络 | 346 |
| `latent_cycle_conv` | 原编码器加局部循环轴残差卷积对照 | 344 |

卷积对照对同样的循环序列使用两层一维卷积，卷积核长度均为 3，隐藏通道数为 49。
同样保留网格、共享参数并零初始化输出层；该模块的理论循环方向感受野为 5 个中间位置。
它用于区分全序列、位置相关的前馈处理与参数量接近的局部卷积处理。

模型还支持 100 循环输入：注入位置是 `[批量, 64, 50, 62]`，最终为 775 个局部特征。
已准备的训练及评价脚本只针对本轮 MIX-20；100 循环输入只有形状及权重恢复检查，不代表已经完成 MIX-100 实验。

## 本轮固定协议

复用服务器 `/root/autodl-tmp/BatLiNet-context-grid-v1` 下已经完成的同条件原模型训练、六通道特征缓存和参考协议。
上下文模型缓存仅用于核对电池划分与标签；本轮输入和前向计算不使用上下文模型的三通道特征。

- 原 207 个训练电池分为 166 个训练、41 个验证，147 个测试电池不变。
- 标签取对数后，用这 166 个训练标签的均值与总体标准差标准化。
- 训练每个目标抽取 2 个训练参考；验证和测试各使用固定 32 个训练参考。
- 训练平均、评价统一对完整 32 个参考取 PyTorch 中位数，两个分支在标准化标签空间等权融合。
- 1000 轮，批量 8，梯度累积 16；AdamW 学习率 0.001、权重衰减 0.01、梯度裁剪 5；训练使用 bfloat16，评价使用 float32。
- 每 25 轮验证，保存第一次达到最低融合验证 RMSE 的 `best.pt`；不合并验证集重新训练。
- `pilot` 运行种子 0、1、2 的两个新模型；`full` 补齐种子 0–7。完成的训练经核对后跳过，不会覆盖；中断的训练不能直接续跑，脚本会报出不完整目录。

本轮结果只能直接和同一协议下的原模型比较，不能直接与 207 个训练参考池的历史结果混算。

## 服务器命令

从已有服务器工作树获取新分支，在独立工作树中运行：

```bash
conda activate batlinet
cd /root/autodl-tmp/BatLiNet-context-grid-v1
git fetch origin agent/cycle-mixer-latent-cross-attention
git worktree add --detach /root/autodl-tmp/BatLiNet-cycle-mixer origin/agent/cycle-mixer-latent-cross-attention
cd /root/autodl-tmp/BatLiNet-cycle-mixer
bash scripts/run_mix20_cycle_mixer.sh /root/autodl-tmp/BatLiNet-context-grid-v1 pilot
```

输出目录为 `workspaces/mix20_cycle_mixer_v1`。
脚本复用原模型对应种子的已完成训练，只训练两个新模型，最后汇总验证结果，不计算测试指标。
验证汇总保存在 `evaluations/mix20_cycle_mixer_v1/validation_summary.json`。

补齐八种子：

```bash
bash scripts/run_mix20_cycle_mixer.sh /root/autodl-tmp/BatLiNet-context-grid-v1 full
```

固定方案并完成八种子训练后，单独计算三组测试结果：

```bash
python -B scripts/evaluate_cycle_mixer_suite.py --seeds 0 1 2 3 4 5 6 7 --run-test
```

会先核对全部 24 次训练的完成状态、最佳验证轮次、缓存和参考名单指纹，再计算测试指标。
测试结果保存于 `evaluations/mix20_cycle_mixer_v1`，包含自身分支、32 个单参考预测、参考聚合结果、实际参考索引和电池编号。

## 本地检查

```bash
python -B scripts/test_cycle_mixer_latent_cross_attention.py
python -B scripts/test_matched_baselines.py
```

检查包括初始化与原模型一致、循环顺序及二维位置保留、两层参数可以学习、局部卷积对照的参数量及作用范围、32 参考分块评价一致性、训练不读取测试特征、检查点恢复，以及评价前拒绝参考名单或最佳轮次不一致的训练。
