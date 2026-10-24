# 软匹配差分的三项独立调整实验

日期：2026-10-24。本文件记录本次已实现的模型、运行协议和操作方法，不记录实验效果或后续方向。未修改主线交接文档。

## 四组配置

| 输出目录中的名称 | 本次改动 |
|---|---|
| `soft_baseline` | 原有单独软匹配潜在差分，重新训练的同期对照 |
| `content_values` | 位置编码只进入匹配的查询和键；目标与参考的值只使用卷积内容 |
| `no_matching_dropout` | 只把注意力概率的随机丢弃率由 0.1 改为 0；其他随机丢弃不变 |
| `pair_supervision` | 原损失外增加权重 0.1 的逐参考差分均方误差 |

后三组分别相对基线只改一项，不互相叠加，均不含循环轴或双轴增强。共享二维卷积编码器、单层四头交叉注意力、两个预测头不变。四组同一种子具有相同的初始参数和初始化后的随机状态；MIX-20 均为 170210 个参数，MIX-100 均为 209890 个参数。改变随机丢弃会改变后续随机状态，因此不声称四组训练中所有随机丢弃掩码相同。

记卷积内容为 F、可学习位置为 P。内容值组计算：

```text
Q = Wq LNq(F_target + P)
K = Wk LNkv(F_reference + P)
A = softmax(Q K^T / sqrt(head_dim))
D = Wv LNkv(F_target) - Dropout(A) Wv LNkv(F_reference)
```

线性层均保留原偏置，目标和参考共用值归一化及投影。输出仍通过原输出投影、前馈残差和参考预测头。单电池预测头仍接收带位置的词元。训练时此组保留 0.1 的匹配概率随机丢弃；评估时关闭。位置仍可通过匹配影响差分，不等于完全删除位置信息，也不主张实现了物理状态对齐。

逐参考组使用标准化自然对数标签 z，损失为：

```text
L = 0.5 MSE(y_own, z_target)
  + 0.5 MSE(mean_r(y_reference), z_target)
  + 0.1 mean_target,reference[(predicted_delta - (z_target-z_reference))^2]
```

实现中的最后一项等价于 `0.1 mean[(y_reference-z_target)^2]`。0.1 为本轮固定值，未用测试集选择。推理融合不变。

## 冻结协议

两任务各四组，各种子 0—7，固定 1000 轮，共 64 次完整训练。先完成 MIX-20 四组全部 32 次训练，再做该任务最终预测；然后同样执行 MIX-100。无验证集选轮，无训练中测试集评估。

- MIX-20：207 个训练电池、147 个测试电池；20 循环，第 11 循环置零；寿命阈值 90%；逻辑和实际训练批量均为 128。
- MIX-100：205 个训练电池、137 个测试电池；100 循环；寿命阈值 80%；逻辑批量 128，实际分批 16，按样本数加权累积梯度。
- 两任务均为六通道、容量轴 1000 点；复用历史输入、标签变换、名单与各种子的固定 32 个测试参考，逐项校验。训练两个参考，测试合并全部 32 个预测后取 PyTorch 较小中位值；MIX-100 按 8 个参考分块计算。
- 单精度、AdamW、学习率 0.001、权重衰减 0.01、不打乱样本。两个分支等权监督，最终在标准化对数空间等权融合，再还原循环寿命。
- **本次四组统一采用独立的训练参考采样序列**：CPU 生成器种子为 `seed + 1000003 * epoch`，一次生成全轮每个样本的两个参考。四组相同种子、相同轮次的参考配对严格一致，不受匹配随机丢弃开关影响。允许自参考，沿用原抽样范围。每 100 轮仍保留旧监测随机数消耗补偿，不计算测试指标。
- 上述独立采样区别于旧实验的全局训练设备随机数采样。因此本轮基线是相同模型结构的新协议对照，不承诺复现旧结果数值；历史结果不混入同期均值。

新入口复用既有历史审计、数据缓存、分块预测和断点保存逻辑。旧组合入口仅新增可选的轮训练回调，并从运行绑定读取模型显示名称；默认训练计算保持不变。

## 服务器启动

在现有服务器工作树中执行：

```bash
cd /root/autodl-tmp/BatLiNet-cycle-mixer && \
conda activate batlinet && \
git switch agent/cycle-mixer-latent-cross-attention && \
git pull --ff-only origin agent/cycle-mixer-latent-cross-attention && \
bash scripts/start_mix_soft_difference_refinements_nohup.sh
```

启动脚本内部调用 `nohup`，不需要外层再加 `nohup` 或 `&`。关闭本地终端不影响后台任务，服务器须持续运行。默认历史目录为 `/root/autodl-tmp/BatLiNet-main3` 和 `/root/autodl-tmp/BatLiNet-main2`；可依次作为启动脚本两个位置参数传入。使用当前已激活环境的 `python`，也可通过 `PYTHON_BIN` 指定解释器。

查看日志：

```bash
tail -f "$(cat workspaces/mix_soft_difference_refinements_v1/latest_log.txt)"
```

`Ctrl+C` 仅退出日志查看。后台进程持续持锁，同一输出目录拒绝重复启动；Python 入口另有队列锁。脚本不停止其他实验，也不保证其他目录的任务不占用同一 GPU。两秒存活检查不代表预检已通过，需看日志。

如只希望先验证真实数据、前向、反向和显存：

```bash
python -u -B scripts/run_mix_soft_difference_refinements.py --prepare-only
```

正式入口要求 Linux 和 CUDA，不自动退回 CPU。四组两任务的训练电池预检全部通过才开始正式队列。每 25 轮保存权重、优化器和随机状态；重启同一命令会核验并续跑，最多重做未保存的 24 轮。恢复要求提交、源码、配置、数据与环境一致，不能中途更新代码后混用原目录。

## 输出与验证

统一输出到 `workspaces/mix_soft_difference_refinements_v1/`：

- `queue.json`：代码提交、源码指纹、环境和八份完整配置。
- `datasets/`、`protocols/`：冻结数据、来源回执和逐种子实际参考索引。
- `preflight.json`、`status.json`、`logs/`：预检、状态及后台日志。
- `<任务>/<组名>_seed<种子>/`：`train.jsonl`、`run.json`、`progress.pt`、`final.pt`、`test.pt`。
- `per_seed.csv`、`per_seed.json`：64 组完整结果后包含全部逐种子指标。
- `summary.json`：每任务每组八种子的均值与样本标准差。
- `paired_comparisons.json`：三项调整分别相对同期基线的配对均值差及获胜种子数；不自动给出显著性或有效性结论。

除最终 RMSE、MAE、MAPE、ACC15，额外汇总单电池分支 RMSE、参考聚合分支 RMSE、标准化对数空间的逐参考均方误差。预测文件保存各分支、32 个参考预测、真值与索引，可独立复算。

本地检查命令：

```bash
python -B scripts/test_mix_soft_difference_refinements.py
python -B scripts/test_mix_cycle_soft_difference.py
python -B scripts/test_soft_aligned_difference.py
bash -n scripts/start_mix_soft_difference_refinements_nohup.sh
```

本地测试覆盖内容差分公式、参数和初始化一致性、真实输入尺寸、随机丢弃范围、逐参考损失、梯度、相同参考序列、四组中断恢复一致性、结果篡改拒绝、分块预测和完整 64 次队列顺序。合成测试通过不等于真实数据精度改善；正式训练和 CUDA 显存检查在服务器运行。
