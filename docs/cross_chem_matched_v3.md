# 跨化学体系三方法统一对照 v3

实现日期：2026-10-24。本文记录本版本的代码设置与运行方式；尚无正式训练结果。旧交接资料和旧结果不改写。

## 入口与输出

- 配置：`configs/cross_chem_matched_v3/protocol.yaml`。
- 主入口：`scripts/run_cross_chem_matched.py`。
- 名单审计与导入：`scripts/cross_chem_matched_data.py`。
- 模型、按更新次数训练、恢复及推理：`scripts/cross_chem_matched_training.py`。
- 启动脚本：`scripts/start_cross_chem_matched_nohup.sh`，内部调用 `nohup`。
- 合成检查：`scripts/test_cross_chem_matched.py`。
- 默认只读来源：`workspaces/cross_chem_two_models_v2`。
- 默认新工作区：`workspaces/cross_chem_matched_v3`。

正式流程需要 Linux、CUDA 和 `batlinet` 环境。本地检查使用 CPU 小规模合成数据，不代表已完成服务器正式尺寸显存检查或真实结果复现。

## 只读复用实际名单

读取旧工作区的 `queue.json`、`dataset/manifest.json`、403个电池缓存，以及112份 `protocols/<体系>/train_<n>/seed_<s>.pt`。核对队列、文件SHA256、特征形状、寿命、电池体系、编号顺序、训练/测试不重叠、同种子跨档位测试固定及训练子集嵌套。已存在的旧测试输出还需与实际名单和权重收据一致。

缺失或不一致即停止，不生成替代名单，不重新提取原始数据。新目录只复制名单到JSON并保存来源收据，不复制大体积特征；旧工作区在运行期间以只读共享锁保护，不创建或改写其锁文件。运行所需旧 `.queue.lock` 文件必须存在，且旧训练进程已经结束。

| 体系 | 来源数 | 目标总数 | 每种子测试数 | 目标训练数 |
| --- | ---: | ---: | ---: | --- |
| LCO | 275个LFP | 37 | 21 | 1、2、4、8、16 |
| NCA | 275个LFP | 22 | 14 | 1、2、4、8 |
| NMC | 275个LFP | 69 | 53 | 1、2、4、8、16 |

种子为0—7。未选中的目标候选不加入训练或测试。旧80/20电池对索引只审计完整性，不用于新训练或选权重；新组合训练遍历全部 `n×275` 电池对。

## 三种方法与共同设置

| 标识 | 模型 | 初始化及目标训练 |
| --- | --- | --- |
| `direct` | 组合编码器＋自身回归头，169507参数 | 随机参数，仅选中的n个目标电池监督 |
| `finetune` | 与direct完全相同 | 对应种子的原始LFP预训练权重，全参数微调，新建优化器 |
| `combination` | 循环轴残差＋软匹配差分双分支，211556参数 | 随机参数，目标与全部LFP来源配对，自身/关系损失各0.5 |

单电池网络在构造完整组合后提取编码器和自身头，同种子保留参数的初始值逐元素一致。微调每个任务独立加载原始来源权重，不接续上一个目标任务，不载入旧MIX或旧迁移权重。

共同输入为前100循环、6通道、1000个容量位置；从旧缓存读取后，将绝对值大于10的特征置零。寿命沿用旧缓存的80%容量终点及补标签规则，不声称重新测定失效寿命。

所有阶段使用 `z=ln(T/1000)`，不拟合标签均值或标准差；逆变换为 `T=1000×exp(z)`。组合的关系头预测此空间中的寿命差，加入来源z；推理对275个来源预测取PyTorch中位数，与自身z各占0.5融合，再逆变换。

AdamW、权重衰减0.01、float32；不打乱顺序。单电池目标阶段每次更新使用全部n个目标电池；来源预训练及组合每次更新最多128样本/电池对，采用确定性循环采样，尾端跨回开头继续。组合的配对顺序为来源在外层、目标在内层，正式各档目标数均整除128，因此每次更新中各目标出现次数相同。物理分批8，梯度按该次实际样本数加权。学习率与更新预算由下面的来源开发阶段选择。

## 来源内部开发与冻结预算

本阶段只加载275个LFP的特征作为开发数据；不加载LCO/NCA/NMC测试数据做开发评价。

- 两个开发划分种子0、1；分别将275个LFP分为219来源、16个开发训练候选、40个开发验证电池，三部分互不重叠。
- 候选学习率为0.001和0.0003；候选更新数为100、300、1000。
- 对每个学习率/划分，在219来源上预训练到1000步，在三个候选步数评价40个验证电池。先按两划分的平均对数寿命均方误差选择共同预训练步数。
- 对每个学习率/划分/开发目标数1或16，运行direct、finetune、combination，各训练到1000步，在三个候选步数评价同一批40个验证电池。微调加载该学习率选出的来源候选权重；组合只用219来源作为开发参考。
- 对两个划分、两个训练数及三种方法等权平均验证误差，选择共同目标更新数；再选择误差最低的学习率。平局依次采用配置中的学习率顺序、较小步数。
- 共28次开发训练、84条候选验证记录。候选缺失、重复或非有限时停止，不排除坏种子或候选。

选定设置保存在 `selection.json`；所有正式方法使用同一学习率和同一目标阶段更新次数。开发评价回调隔离随机状态并恢复模型模式，不改变后续训练的随机失活序列。直接训练的正式权重不在LFP上做梯度训练，但其共同超参数规则和其他方法一样由LFP开发阶段确定。

本版本采用固定最终更新权重，不做关系验证早停，也不根据目标测试结果选轮次。相同更新次数不等同于相同计算量；逐步日志另记录样本暴露数及实际训练耗时。

## 正式训练、测试与恢复

先运行8个种子的全275 LFP预训练，再运行14档×8种子×3方法共336次目标训练，正式合计344次。全部正式训练及权重收据完整通过核验后，才统一运行336组目标测试；开发28次训练另计。

每25步及最终步原子保存 `progress.pt`，包含模型、优化器、全部随机状态、采样游标、累计更新数和数据/代码/环境/预算绑定。中断后重复同一启动命令，从最近持久化状态恢复；已经完整完成的训练和测试经核验后复用。绑定发生变化时拒绝混合恢复，不覆盖为另一轮实验。

每个运行目录含 `run.json`、`train.jsonl`、`progress.pt`、`final.pt`；目标任务测试后增加 `test.pt`、`evaluation.json`。根目录保存 `queue.json`、`protocol_import_receipt.json`、`preflight.json`、`selection.json`、`status.json`、逐种子CSV/JSON及 `summary.json`。

测试保存全部电池真值、预测、自身及关系诊断、实际参考顺序和权重指纹；报告八种子均值及样本标准差，不剔除高误差种子。RMSE/MAE单位为循环，CSV/JSON中的MAPE/ACC15为小数比例，打印报告时乘100显示百分数。

## 服务器命令

在原实验工作树执行，拉取后由启动脚本激活环境并后台运行：

```bash
cd /root/autodl-tmp/BatLiNet-cycle-mixer
git pull --ff-only origin agent/cycle-mixer-latent-cross-attention
bash scripts/start_cross_chem_matched_nohup.sh
```

启动脚本内部已有nohup；无需在外层再包一次。默认旧目录缺失时停止。若旧完整工作区在其他位置，显式指定：

```bash
bash scripts/start_cross_chem_matched_nohup.sh \
  --existing-workspace /实际路径/cross_chem_two_models_v2
```

仅审计实际名单并执行正式尺寸前向、反向及推理预检，不启动来源开发和正式训练：

```bash
bash scripts/start_cross_chem_matched_nohup.sh --prepare-only
```

仅执行来源开发并锁定预算，暂不进行正式344次训练：

```bash
bash scripts/start_cross_chem_matched_nohup.sh --pilot-only
```

完成准备或开发后，执行不带模式参数的启动命令即可继续。后台进程运行时不要重复启动；可关闭本地终端，但服务器必须保持运行。

```bash
tail -f "$(cat workspaces/cross_chem_matched_v3/latest_log.txt)"
cat workspaces/cross_chem_matched_v3/status.json
bash scripts/start_cross_chem_matched_nohup.sh --report
```

报告命令前台只读执行，不创建或改写结果。每25步输出当前运行损失、样本暴露数和预计剩余分钟；正式队列按已完成运行耗时给出粗略剩余训练时间，未包含全部测试耗时。

## 本地检查命令

```powershell
& 'D:/GongZuo/Anaconda/azb/envs/batlinet/python.exe' -X utf8 -B scripts/test_cross_chem_matched.py
& 'D:/GongZuo/Git/azb/bin/bash.exe' -n scripts/start_cross_chem_matched_nohup.sh
```
