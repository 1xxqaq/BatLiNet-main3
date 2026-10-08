# 循环轴残差与软匹配潜在差分组合：MIX双任务

日期：2026-10-08。用户决定暂缓MATR-1、HUST、MATR-2，当前只训练组合版。MIX-20与MIX-100各种子0—7、固定1000轮，共16次；不补训对照。本地已完成代码和合成预检，服务器正式训练尚未启动。

## 组合方式

在 `SoftAlignedDifferenceBatLiNetRULPredictor` 的共享编码器中，复用已有循环轴残差模块，插在第三层卷积激活后、最后池化前；目标和参考共用参数。循环轴网络隐藏宽度16，输出层零初始化、初始化隔离随机数，因此同种子初始输出与仅软匹配差分版逐项一致。中间循环序列长度MIX-20为10、MIX-100为50，不增加原始输入循环。

编码后仍使用已有单层四头软匹配：目标和参考经同一值投影，计算目标值减去注意力匹配的参考值，再使用原输出投影和前馈残差。没有新增注意力层、额外损失或预测头，也没有把原始六通道先作电池间差分。

| 项目 | MIX-20 | MIX-100 |
|---|---:|---:|
| 完整有效训练／测试电池 | 207／147 | 205／137 |
| 早期输入循环数 | 20，第11循环置零 | 100 |
| 寿命终点 | 额定容量90% | 额定容量80% |
| 组合参数量 | 170556 | 211556 |
| 相对仅软匹配差分新增参数 | 346 | 1666 |
| 最终词元数 | 155 | 775 |
| 逻辑／物理批量 | 128／128（尾批79） | 128／16（尾逻辑批77，末分批13） |
| 测试参考分块 | 一次32个 | 分块8个，合并32个后取较小中位值 |

保持六通道、容量轴1000点、关闭循环过滤并保留原边缘／毛刺清理；标签直接复用历史标准化自然对数标签及训练均值、样本标准差，不重算或舍入训练标签。训练2参考，测试固定32参考，alpha0.5；单精度，AdamW学习率0.001、权重衰减0.01；不打乱、不新增梯度裁剪，不使用即时编译。

训练参考沿对应旧入口使用训练设备的全局PyTorch随机数；每100轮只推进旧监测的随机数消耗，不读取测试输入、标签或指标。MIX-20整批训练、MIX-100样本加权分批流程已用同一组合模型的小尺寸合成数据与旧入口逐项核对：一轮权重及监测后的随机状态相同。这不能证明跨环境的正式结果逐位相同。GPU后端设置沿既有设种子入口，TF32等实际值写入运行环境，不强行修改为三任务入口的策略。

## 历史数据与参考

MIX-20用既有原始／基础潜在正式预测中的数据和独立固定协议进行全八种子核对；MIX-100用基础潜在正式预测及独立协议（若缺失则核对并恢复预测实际记录的索引）。默认历史目录为 `/root/autodl-tmp/BatLiNet-main3`，MIX-20原始目录为 `/root/autodl-tmp/BatLiNet-main2`，路径沿用已成功运行的软匹配实验。

两任务数据及参考全部核验后，使用训练电池完成组合模型前向、反向和测试分块显存预检，均通过才正式训练。不重新抽取测试参考，不使用历史训练权重初始化组合。先训练MIX-20全部八种子，再测试；之后同样处理MIX-100。

如果既有软匹配及循环轴结果在标准工作区中，入口可只读核验历史预测、真值、数据身份、实际参考、指标、最终权重指纹及结构绑定，把完整八种子的参考组单独汇总。缺失或不符的参考组会明确登记并排除，不妨碍新的组合训练；不推断不存在的MIX-100软匹配结果，也不混入其他输入批次。历史参考与新组合结果分文件保存；它们不是本次新增同条件对照，不能据一次比较证明两个模块存在协同收益。

## 服务器后台启动

```bash
cd /root/autodl-tmp/BatLiNet-cycle-mixer && \
conda activate batlinet && \
git switch agent/cycle-mixer-latent-cross-attention && \
git pull --ff-only origin agent/cycle-mixer-latent-cross-attention && \
bash scripts/start_mix_cycle_soft_difference_nohup.sh
```

脚本内部用nohup后台启动，关闭标准输入、合并输出和错误、无缓冲写独立日志。可以关闭本地电脑、终端或断网；服务器必须运行。启动后两秒的进程存在检查不等于数据及显存预检通过，查看日志确认。启动器和入口会拒绝重复组合队列；发现三任务队列仍持锁时，提示先停止，不擅自终止服务器进程。

查看进度：

```bash
tail -f "$(cat /root/autodl-tmp/BatLiNet-cycle-mixer/workspaces/mix_cycle_soft_difference_v1/latest_log.txt)"
```

按Ctrl+C只退出查看，不停止训练。所有输出在全新目录 `workspaces/mix_cycle_soft_difference_v1/`，不会覆盖旧MIX实验。故障详情写日志与status.json；不自动改分批、改精度、跳过坏种子或无限重试。恢复需相同代码提交、配置、数据和环境，重新执行同一启动命令即可。每25轮保存恢复用权重、优化器及Python／NumPy／PyTorch／当前GPU随机状态；最多重做未保存的24轮，超出检查点的中断日志先归档。

## 输出与结果查看

- `queue.json`：本次提交、源码指纹、环境、两份完整配置。
- `datasets/<任务>/data.pt`、`receipt.json`：历史输入、标准化标签、循环单位真值、名单、标签统计和来源。
- `protocols/<任务>/seed_*.pt`：复用的历史实际32参考及电池顺序。
- `preflight.json`、`status.json`、`logs/`：真实数据／显存预检与后台运行状态。
- `<任务>/latent_cycle_soft_difference_seed*/train.jsonl`、`run.json`、`progress.pt`、`final.pt`、`test.pt`：每种子日志、最终权重、恢复状态和预测诊断。`final.pt`只含模型及绑定；优化器留在`progress.pt`。诊断为标准化自然对数空间，寿命预测与真值另存循环单位。
- `per_seed.json`、`per_seed.csv`：16份新组合逐种子指标。
- `summary.json`：每任务新组合八种子均值和样本标准差。
- `historical_reference_summary.json`及`datasets/<任务>/historical_reference_receipts.json`：通过绑定核验的既有参考及缺失／拒绝原因，不与新结果混算。

```bash
cd /root/autodl-tmp/BatLiNet-cycle-mixer
cat workspaces/mix_cycle_soft_difference_v1/status.json
python - <<'PY'
import json
from pathlib import Path
root=Path('workspaces/mix_cycle_soft_difference_v1')
for file,title in [('summary.json','本次组合版'),('historical_reference_summary.json','既有参考（非新增同条件对照）')]:
    path=root/file
    print('\n'+title)
    if not path.exists():
        print('尚未生成'); continue
    for row in json.loads(path.read_text()):
        print(row['task'],row['model'])
        for metric,value in row['metrics'].items():
            print(f"  {metric}: {value['mean']:.6f} ± {value['sample_std']:.6f}")
PY
```

均方根误差（RMSE）、平均绝对误差（MAE）单位为循环；平均绝对百分比误差（MAPE）、15%以内准确率（ACC15）为小数比例。指标用双精度归约，准确率条件相对误差≤0.15。固定最终权重，不按测试指标挑种子、轮次或配置。

## 本地检查与限制

新增七项CPU合成检查通过：正式结构参数量／零初始化／随机隔离、两个输入及差分投影和循环轴梯度、与两旧训练入口及监测随机数的对齐、检查点恢复逐项相等、完成项复用／错误绑定拒绝、全32参考与分块及现有模型预测一致、历史缓存／参考修改拒绝、可选历史参考绑定核验及异常排除。七项测试按组合分组，不代表七项正式实验。记录见 `docs/mix_cycle_soft_difference_self_test.json`。

原软匹配差分六项回归检查通过；后台脚本通过Bash语法检查。所有训练检查只用临时合成数据。本地没有CUDA，正式历史数据的本次服务器核验及组合显存验收尚待启动执行。MIX-20整批显存不足时停止，不静默减批量，以免改变已声明的比较协议。
