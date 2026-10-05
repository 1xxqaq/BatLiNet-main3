# MIX-20：207块全训练的双轴交互残差探索实验

## 实验入口与模型

入口：`scripts/run_mix20_axis_interaction_207.py`。
模型代码：`src/models/rul_predictors/axis_interaction_latent_cross_attention_batlinet.py`。

三组新增实验均使用种子0–7，从头训练全部参数：

| 训练名称 | 编码器处理 | 参数量 |
| --- | --- | ---: |
| `latent_dual_axis_mixer` | 现有双轴模块直接相加：`G + Rc(G) + Rq(G)` | 172618 |
| `latent_axis_within_fusion` | 各视角内部注意力，逐位置融合后补充到循环视角 | 221610 |
| `latent_axis_cross_fusion` | 双向跨视角交叉注意力，逐位置融合后补充到循环视角 | 221610 |

另外只读核对并复用已完成的207块原latent重训和循环轴增强版八种子结果。
原始BatLiNet与历史latent的正式预测作为历史参照，最终总表有56行。
已有循环轴增强版未保存完整软件和硬件环境信息；新增三组在本次环境下比较，不声称完全重建旧增强版的环境。

## 结构

共享卷积仍在最后一次池化之前生成 `G: [B,64,10,62]`。
循环轴和容量轴模块继续分别使用 `10→16→10`、`62→16→62` 的两层网络。
两分支均读取G，得到 `C=G+Rc(G)` 与 `P=G+Rq(G)`，保持同一二维网格。

注意力融合版本的步骤如下：

1. 循环视角沿容量方向取均值，得到10个循环位置词元；容量视角沿循环方向取均值，得到62个容量位置词元。汇聚只供交互路径使用，C和P的局部信息仍直接送入融合网络。
2. 两组词元分别加可学习轴向位置参数，再进行层归一化。注意力宽度64，四个头，新增注意力不使用随机丢弃。
3. 跨视角版本：循环词元查询容量词元，容量词元查询循环词元；内部注意力对照：每组词元查询自身。两版本参数、初始权重和随机数消耗完全一致，差别只有键和值来自哪一组词元。
4. 每个二维位置拼接C、P、该循环位置的交互结果、该容量位置的交互结果。通过 `层归一化→线性256→32→GELU→线性32→64` 生成补充量Δ。
5. 输出 `C+Δ`，再执行原有最后池化、随机丢弃、二维展开和位置编码，保持155个64维局部词元。

两轴模块输出层及融合网络最后一层均为零初始化。新增初始化在独立随机数上下文中完成，不改变原卷积、预测头或后续参考抽样的初始随机数状态。
初始输出等同于循环轴增强版；融合输出层先获得梯度，后续步骤允许注意力和容量路径获得梯度。CPU检查覆盖了该过程，未发生多层零初始化导致路径无法学习的问题。

这是**非对称的探索结构**：交叉注意力双向交换信息，但最终残差跳连使用循环视角。不能将其描述为两个视角完全对称的融合模型，也不能在没有正式结果时声称互补有效。
这一交互思路参考了Axial-CrossViT的跨分支选择性查询；本实现保留二维局部网格，不是该论文全局汇总词元模型的复现。
论文：<https://ssrn.com/abstract=7177059>。

原六通道输入、清理方法、目标与参考电池的交叉注意力、寿命预测头、训练损失及参考聚合均保持原实现。
该实现也支持MIX-100输入形状：775个局部词元，两种注意力融合版本参数量均为265170。MIX-100只进行了结构检查，不包含正式训练结果。

## 207块固定训练与测试协议

- 使用历史完整数据包中的207个训练电池及147个测试电池，保留电池顺序、六通道特征和训练标签自然对数标准化统计量。
- 不从166/41缓存拼接新数据，不重新生成固定参考名单。
- **不设验证集。固定训练1000轮，使用第1000轮权重；不早停，不依据测试指标选择轮次。**
- 沿用历史训练函数：批量128，不打乱顺序，完整32位浮点精度，梯度累积1，AdamW学习率0.001、权重衰减0.01，无新增梯度裁剪。
- 每目标随机抽取两个训练参考。每100轮只模拟历史监控的随机抽样以对齐随机数状态，不计算测试预测或测试指标。
- 所有24次新增训练完整通过核对后，自动按八份旧固定名单测试；每目标32个训练参考，使用原预测器的中位数聚合及等权分支预测。
- 输出RMSE、MAE、MAPE和ACC15的八种子均值、样本标准差；显示跨视角版本相对循环轴、直接相加、内部注意力对照的逐种子平均差及胜出种子数。
- MAPE和ACC15均为小数比例，ACC15为0.60表示60%。

模型优劣需依据同一207块协议的结果比较，不能把此次测试结果直接与166/41协议当作同条件比较。

## 完成核对、保存与重跑

训练前核对历史16份预测、现有循环轴八份完整训练及测试、现有原latent重训八份完整训练及测试。
核对数据指纹、原配置、实际参考索引、训练代码、训练轮次及保存指标重算。
原latent重训保存的运行环境必须与本次一致；如环境或已有资料不匹配，先报错，不开始新增训练。

新增输出独立保存到 `workspaces/mix20_axis_interaction_207_v1/<训练名称>_seed<0–7>/`：

- `run.json`：模型配置、参数量、训练状态、数据/代码/协议/基线文件指纹、运行环境。
- `train.jsonl`：每轮训练损失和从开始训练起的累计秒数，共1000行。
- `epoch1000.pt`：第1000轮权重与来源记录。
- `test.pt`：固定测试预测、寿命单位预测与标签、分支诊断、实际参考名单、指标与权重指纹。
- 工作区根目录 `summary.json`：全部56行结果。

完整训练经核对后跳过，已有测试也核对后复用；不会覆盖旧工作区。
已有训练目录不完整时停止，不支持从中断轮次自动续训。
每20轮显示训练损失、本种子预计剩余分钟数及后续未完成训练次数。时间按当前平均速度估计。

## 服务器完整命令

在现有的服务器工作树中执行以下命令。默认直接进行24次新增训练及最终固定测试，不是试跑。

```bash
conda activate batlinet
cd /root/autodl-tmp/BatLiNet-cycle-mixer
git pull --ff-only origin agent/cycle-mixer-latent-cross-attention
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p logs
nohup python -u -B scripts/run_mix20_axis_interaction_207.py \
  --history-root /root/autodl-tmp/BatLiNet-main3 \
  --original-root /root/autodl-tmp/BatLiNet-main2 \
  --cycle-runs /root/autodl-tmp/BatLiNet-cycle-mixer/workspaces/mix20_cycle_mixer_207_v1 \
  --current-runs /root/autodl-tmp/BatLiNet-cycle-mixer/workspaces/mix20_latent_current_207_v1 \
  --workspace /root/autodl-tmp/BatLiNet-cycle-mixer/workspaces/mix20_axis_interaction_207_v1 \
  > logs/mix20_axis_interaction_207_full.log 2>&1 < /dev/null &
```

查看日志：

```bash
tail -f /root/autodl-tmp/BatLiNet-cycle-mixer/logs/mix20_axis_interaction_207_full.log
```

`nohup`任务在SSH断开后继续运行；按Ctrl+C停止查看日志，不会停止后台训练。服务器重启或训练进程报错仍会停止任务。

只读核对现有资料而不训练：

```bash
python -B scripts/run_mix20_axis_interaction_207.py --audit-only
```

## 本地验证

```bash
python -B scripts/test_axis_interaction_207.py
python -B scripts/test_cycle_mixer_207.py
python -B scripts/test_latent_current_207.py
```

新增七项CPU检查及旧协议十项回归检查通过。覆盖参数量与同种子初始化、二维词元数量、跨视角依赖、梯度流通、随机数消耗、24次训练完成后才测试、旧参考名单、重跑复用、损坏/不完整资料拒绝及只读核对。
流程检查使用小型数据的一轮真实CPU训练及模拟完成记录；它们不是207块电池的正式训练或准确率结果。
