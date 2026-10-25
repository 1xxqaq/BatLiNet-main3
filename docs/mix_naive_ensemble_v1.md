# MIX 同步训练双编码器朴素集成

本批新增一个不共享参数的双编码器朴素集成模型。MIX-20、MIX-100 各用种子 0—7，从头训练 1000 轮，共 16 次训练。自身子网和参考关系子网在同一训练循环内同步更新，两个子网的参数完全独立；与完整组合模型的共享编码器结构不同。本次输出保存在独立工作区，不修改已有分支消融结果。

## 模型及比较含义

自身子网由循环轴残差编码器和自身预测头构成。参考关系子网包含另一套编码器、软匹配潜在差分和关系预测头；该子网内部的目标与参考仍共用编码器。两个子网之间没有共享参数，也不交换中间特征。

每个逻辑批量计算 `0.5 × 自身损失＋0.5 × 参考损失`，由同一个 AdamW 优化器更新两个子网。测试分别得到自身预测和固定参考聚合预测，再在历史标准化自然对数寿命空间等权融合，最后还原为循环数。损失系数和预测融合分别控制。

由于两部分损失各自只依赖对应子网，在初始化、数据顺序、参考抽样、优化器更新及各分支随机数序列一致时，同步训练与分别训练后集成在数学上等价。这组实验用于核验这一等价关系及实际实现，不把同步运行本身解释成新的协同机制。

本实现为两个分支分别维护随机失活所用的随机状态，避免一个分支的前向计算改变另一个分支的随机序列。与旧结果比较时仍需核对实际运行条件，不能仅凭“同种子”承诺跨环境数值逐位一致。

## 固定实验协议

| 项目 | MIX-20 | MIX-100 |
| --- | --- | --- |
| 有效训练／测试电池 | 207／147 | 205／137 |
| 输入 | 六通道，前 20 循环，第 11 循环置零 | 六通道，前 100 循环 |
| 每循环容量采样点 | 1000 | 1000 |
| 寿命终点 | 额定容量的 90% | 额定容量的 80% |
| 参数总量 | 290614 | 372614 |
| 种子及轮数 | 种子 0—7，固定 1000 轮 | 同左 |
| 逻辑／物理批量 | 128／128 | 128／16 |
| 训练／测试参考数 | 2／32 | 同左 |
| 精度及优化器 | 单精度，AdamW，学习率 0.001，权重衰减 0.01 | 同左 |

沿用历史输入清理、电池划分、标准化自然对数标签和固定测试参考。训练电池不打乱，参考从训练池有放回抽取，使用种子加 310000 的独立 CPU 随机数生成器；训练两个参考取均值，测试 32 个参考取中位数。训练不读取测试指标，不使用验证集挑轮。

先核对两个任务的数据和全部参考协议，并以训练数据完成真实显存、前向和梯度预检。所有 16 次训练完成并核对最终检查点后，再进行最终测试。每 25 轮保存模型、优化器及恢复所需随机状态；中断后重新运行相同启动命令，从最近保存的检查点恢复。已有输出的源码、提交、配置、数据、参考及运行环境绑定不符时拒绝恢复。

## 先建立独立服务器代码目录

旧分支消融正在运行时，不要在其代码目录执行 `git pull`、切换分支或修改源文件。旧入口将代码提交及源文件指纹绑定到检查点，修改后可能无法继续或恢复旧实验。

首次部署时，在旧仓库只获取远程提交，再建立独立代码工作树。以下命令不改变旧目录的检出提交及源文件：

```bash
cd /root/autodl-tmp/BatLiNet-cycle-mixer
git fetch origin agent/cycle-mixer-latent-cross-attention
git worktree add --detach /root/autodl-tmp/BatLiNet-naive-ensemble \
  origin/agent/cycle-mixer-latent-cross-attention
cd /root/autodl-tmp/BatLiNet-naive-ensemble
conda activate batlinet
bash scripts/start_mix_naive_ensemble_nohup.sh
```

工作树目录已存在时，不重复执行 `git worktree add`。后续查看或恢复直接进入这个新目录，重新运行启动脚本即可。

新入口取得自身输出队列锁后，会等待旧分支消融使用的同一 GPU 锁释放，再开始数据预检和训练；不终止旧进程。锁路径由 `--branch-workspace` 的父目录定位为 `gpu_locks/<设备名>/.queue.lock`，例如旧输出目录旁的 `gpu_locks/cuda_0/.queue.lock`。因此独立代码工作树仍会与旧队列正确排队。未参与此共享锁的其他训练入口仍须人工避免同时占用同一 GPU。

## 启动参数与恢复

启动器内部使用 `nohup`，关闭标准输入、合并日志，并通过 Python 的 `-B` 关闭字节码缓存写入。无需外层再加 `nohup` 或 `&`。启动两秒后检查进程是否存活；进程存活可能表示仍在等待 GPU 锁，不等于已通过预检或开始训练。可关闭终端、断开连接或关闭本地电脑，服务器须保持运行。

默认路径如下：

| 参数 | 默认值 |
| --- | --- |
| 历史目录 | `/root/autodl-tmp/BatLiNet-main3` |
| 原始目录 | `/root/autodl-tmp/BatLiNet-main2` |
| 新输出目录 | 当前代码根目录下的 `workspaces/mix_naive_ensemble_v1` |
| 旧分支消融输出目录 | `/root/autodl-tmp/BatLiNet-cycle-mixer/workspaces/mix_branch_ablation_v1` |
| 解释器／设备 | `python`／`cuda:0` |

可通过四个位置参数覆盖路径，通过环境变量指定解释器及设备：

```bash
PYTHON_BIN=python DEVICE=cuda:0 bash scripts/start_mix_naive_ensemble_nohup.sh \
  /root/autodl-tmp/BatLiNet-main3 \
  /root/autodl-tmp/BatLiNet-main2 \
  /root/autodl-tmp/BatLiNet-naive-ensemble/workspaces/mix_naive_ensemble_v1 \
  /root/autodl-tmp/BatLiNet-cycle-mixer/workspaces/mix_branch_ablation_v1
```

若旧分支消融使用了其他工作区，必须同步修改第四个参数，保证两个入口使用同一 GPU 锁目录。旧工作区仅用于只读对照及定位共享锁，新入口不向旧实验结果写入内容。旧结果暂缺或未完成，不妨碍本次 16 次训练。

只准备数据、参考协议并完成真实显存及梯度预检，不启动正式训练：

```bash
python -B -u scripts/run_mix_naive_ensemble.py --prepare-only
```

该命令也等待共享 GPU 锁，支持 `--history-root`、`--original-root`、`--workspace`、`--branch-workspace`、`--device`。正式准备及训练须使用有 CUDA 的 Linux 服务器。

## 进度及结果查看

```bash
cd /root/autodl-tmp/BatLiNet-naive-ensemble
tail -f "$(cat workspaces/mix_naive_ensemble_v1/latest_log.txt)"
```

在 `tail -f` 中按 Ctrl+C 只退出查看，不停止训练。启动器用文件锁拒绝重复启动，不会将系统复用的旧进程号误认为正在运行的队列。入口错误会停止并写入日志和状态，不自动缩小批量、切换精度或跳过种子。

```bash
cat workspaces/mix_naive_ensemble_v1/status.json
python -B scripts/run_mix_naive_ensemble.py --report
```

`--report` 只读取已保存结果，不启动训练、不推理，也不要求 CUDA；其他输出目录通过 `--workspace` 指定。新工作区保存来源和运行绑定、历史数据及参考协议、预检与状态、逐轮日志、恢复检查点、最终权重、逐电池预测、逐种子指标和汇总。

旧 48 次分支消融结果可用后，入口只读核验并作为单独对照记录；缺失或核验不通过会明确记录，不将不符结果混入新实验。比较时区分完整共享模型、重训单分支、旧独立集成和本次同步训练朴素集成，不将不同训练批次的均值直接当作模块因果收益。

`independent_equivalence.json` 记录旧单分支的代码、环境、配置、数据和参考序列核验，以及两子网最终参数是否逐位一致、与旧独立集成预测的最大差值。完整模型也核验通过时一并纳入比较。`paired_previous_comparisons.json` 仅汇总同任务完整八种子的配对差和改善种子数；缺失或核验失败的对照不补数，不从总体均值反推逐种子结果。`--report` 控制台比例指标显示为百分数，JSON/CSV 中保存小数比例。

指标包括均方根误差（RMSE）、平均绝对误差（MAE）、平均绝对百分比误差（MAPE）和相对误差 15% 以内准确率（ACC15），汇总八种子均值与样本标准差。RMSE、MAE 单位为循环，MAPE、ACC15 为比例。
