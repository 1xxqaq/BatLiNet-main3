# 双轴残差编码器：MIX-20 八种子对照

## 模型结构

本轮在现有循环轴残差增强版上加入容量轴分支，保留原六通道输入、输入清理、三层二维卷积、位置编码、共享编码器、交叉注意力和两个寿命预测头。

第三层卷积和激活后的特征网格为 `[批量, 64, 10, 62]`，两个分支并行读取这张网格：

```text
六通道曲线 → 原共享二维卷积 → 网格 G
                               ├─ 循环轴：10 → 16 → 10 → 修正量
                               └─ 容量轴：62 → 16 → 62 → 修正量
                 G + 循环轴修正量 + 容量轴修正量
                               ↓
               原池化、位置编码 → 155个64维局部特征
                               ↓
                  原交叉注意力与两个预测头
```

循环轴分支在每个容量位置和特征通道共享参数；容量轴分支在每个循环位置和特征通道共享参数。目标和参考电池使用同一个编码器。
两个分支都使用 GELU 激活，输出层权重和偏置初始化为零。辅助模块初始化隔离随机数状态，没有新增随机丢弃层。
容量轴直接读取原网格，不读取循环轴增强后的网格；完整二维位置和特征数量保留。
这里的轴位置已经经过卷积与池化，不能解释成独立原始循环或独立原始采样点。

| 训练入口名称 | 结构 | MIX-20 总参数 | MIX-100 总参数 |
| --- | --- | ---: | ---: |
| `latent_cycle_mixer` | 原卷积＋循环轴；复用已有结果 | 170556 | 211556 |
| `latent_capacity_mixer` | 原卷积＋容量轴；从头训练 | 172272 | 211952 |
| `latent_dual_axis_mixer` | 原卷积＋并行双轴；从头训练 | 172618 | 213618 |

容量分支新增 2062 个参数。关闭容量分支可恢复现有循环轴编码器的计算路径。
这项功能用于结构检查；本轮两个新模型都从头训练，不加载已有训练权重。
MIX-100 的形状检查已覆盖 `[批量, 64, 50, 62]` 和最终775个局部特征，本轮训练入口只运行 MIX-20。
本轮还没有正式实验结果，不能提前认定容量轴带来增益。

## 固定训练和评价协议

- 复用166训练／41验证／147测试划分，以及既有六通道特征缓存；上下文缓存只用于核对电池编号和标签。
- 每种模型使用种子0—7。复用已有循环轴模型八次训练，新增容量轴和双轴各八次，共新增16次训练。
- 标签取自然对数后按166个训练标签的均值和总体标准差标准化。
- 每个训练目标随机抽取2个训练参考，训练聚合取平均。验证和测试复用既有固定32参考名单，在标准化标签空间对完整32个参考取 PyTorch 中位数；两个预测头等权融合后反变换。
- 每次1000轮，物理批量8，累积16个批量；末组按实际样本数加权。AdamW 学习率0.001、权重衰减0.01、梯度裁剪5。训练使用 bfloat16 自动混合精度，评价使用 float32。
- 每25轮验证，严格改善才保存 `best.pt`，采用第一次达到最低融合验证RMSE的轮次；不合并验证集重训。
- 默认命令只训练和汇总验证集。测试入口单独执行，要求全部三组八种子已经完成。

训练循环已与旧循环轴入口做两轮数值对照：同种子检查点权重、损失、验证指标及训练后的随机数状态完全一致，包括末批不足和梯度累积。
这个小规模CPU检查验证实现是否延续旧流程，不代表正式GPU实验已经完成，也不保证两个新模型效果更好。

## 代码和输出位置

| 内容 | 仓库内位置 |
| --- | --- |
| 双轴编码器及注册预测器 | `src/models/rul_predictors/dual_axis_latent_cross_attention_batlinet.py` |
| 复用旧前向路径的适配器 | `src/models/rul_predictors/dual_axis_matched_baselines.py` |
| 完整八种子训练、核对和独立测试入口 | `scripts/run_mix20_dual_axis.py` |
| 本地功能与流程检查 | `scripts/test_dual_axis_latent_cross_attention.py` |
| 新训练日志、配置、最佳权重 | `workspaces/mix20_dual_axis_v1/模型名_seed编号/` |
| 三组24行验证汇总 | `evaluations/mix20_dual_axis_v1/validation_summary.json` |
| 独立测试预测、参考索引和分支诊断 | `evaluations/mix20_dual_axis_v1/模型名/seed编号.pt` |
| 独立测试汇总 | `evaluations/mix20_dual_axis_v1/test_summary.json` |

每次新训练的 `run.json` 记录数据和参考名单指纹、代码提交、相关源文件指纹、运行环境、训练配置和参数量；每轮记录到 `train.jsonl`。
开始前核对八次已有循环轴训练、最佳检查点、数据及全部验证／测试参考名单；不生成或覆盖旧参考协议。
已有新训练只有完成且配置、源代码、数据和运行环境一致时才跳过。未完成目录会报错，不会被覆盖或自动续训。
`--audit-only` 只读核对，不写文件、训练或计算测试指标，可以使用CPU。

原 `latent_cross_attention`、循环轴模型、旧训练脚本和旧实验结果均保留；本轮不改交接文档。

## 服务器命令

在已有独立工作树中更新代码后，直接运行全部八种子：

```bash
conda activate batlinet
cd /root/autodl-tmp/BatLiNet-cycle-mixer
git fetch origin agent/cycle-mixer-latent-cross-attention
git checkout --detach origin/agent/cycle-mixer-latent-cross-attention
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p logs
nohup python -u -B scripts/run_mix20_dual_axis.py \
  --existing-root /root/autodl-tmp/BatLiNet-context-grid-v1 \
  --cycle-runs /root/autodl-tmp/BatLiNet-cycle-mixer/workspaces/mix20_cycle_mixer_v1 \
  > logs/mix20_dual_axis_full.log 2>&1 </dev/null &
tail -f logs/mix20_dual_axis_full.log
```

每20轮以及每次验证显示当前损失、本次训练预计剩余时间、后续训练次数和整批粗略剩余时间，估计包含已观测的验证耗时。
首次估计和跨模型估计会有波动；运行中可用 `Ctrl+C` 退出日志监视。
`nohup` 让训练在终端或网络连接断开后继续运行；服务器关机、进程被终止或训练报错仍会停止。

默认命令结束后先查看验证结果。方案固定后，独立测试命令为：

```bash
python -u -B scripts/run_mix20_dual_axis.py \
  --existing-root /root/autodl-tmp/BatLiNet-context-grid-v1 \
  --cycle-runs /root/autodl-tmp/BatLiNet-cycle-mixer/workspaces/mix20_cycle_mixer_v1 \
  --run-test
```

测试入口先核对全部24次训练，不补跑缺失训练。每个结果保存完整32参考索引、单参考预测、聚合预测和电池编号；已有测试结果会核对来源与指标后复用。

本地检查：

```bash
python -B scripts/test_dual_axis_latent_cross_attention.py -v
```
