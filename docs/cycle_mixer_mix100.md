# MIX-100：原 latent 与循环轴残差增强版同条件八种子对照

入口：`scripts/run_mix100_cycle_mixer.py`。

使用历史 MIX-100 的 205 个训练电池、137 个测试电池，六通道、100 循环、1000 个容量位置，保留历史电池顺序、输入及自然对数/样本标准差标签变换。
启动时只读核对原 latent 八份正式预测中的完整数据、标签、顺序和实际参考名单。
存在原固定协议目录时逐种子核对；若原目录未复制，直接恢复历史预测保存的实际名单，不随机生成新名单。
恢复的相同八份名单保存于新实验的 `protocols` 目录，不修改历史文件。

两个模型都从头训练种子 0–7，共 16 次训练，每次固定 1000 轮，全部完成后统一测试，不使用测试集选择权重或停止轮次。
原模型读取原 MIX-100 配置，参数量 209890；增强版保持已有循环轴残差结构，隐藏宽度 16、输出层零初始化，参数量 211556。
注意力、预测头及其他模型设置不变；本次不训练局部循环卷积对照。

两个模型共用 float32、AdamW、学习率 0.001、权重衰减 0.01、不打乱训练顺序、每个训练目标两个参考、每个测试目标 32 个固定参考。
有效批量保持 128，默认物理分批大小 16；每轮两个逻辑批量分别为 128 和 77 个电池，分别累积 8 次和 5 次前向/反向后更新参数，每轮共更新两次。
按实际样本数加权梯度，最后不足 16 的分批以及 77 个电池的批量均正确归一化。
物理分批改变了历史整批训练的随机失活随机数调度，不能声称与历史整批训练逐位一致；两个新模型使用相同分批设置，以本次重训原模型作为主要对照。
每 100 轮仅推进旧监测所消耗的随机数状态，不读取测试特征或标签。

结果独立保存于 `workspaces/mix100_cycle_mixer_v1`：每个模型/种子含 `run.json`、`train.jsonl`、`epoch1000.pt`、`test.pt`；汇总 `summary.json` 包含历史原模型和两组新实验共 24 行。
记录数据、参考索引、代码及权重指纹、运行环境、逐轮损失、实际更新次数和耗时；每 20 轮显示本次训练预计剩余时间及后续训练数量。
完整训练核对后跳过，不支持从中断轮次续训；已有目录不完整或与当前协议不一致时退出，不覆盖权重。
`nohup` 允许 SSH 断开后继续运行，服务器重启或进程终止仍会中断。

更新当前实验工作树后运行：

```bash
conda activate batlinet
cd /root/autodl-tmp/BatLiNet-cycle-mixer
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p logs
nohup python -u -B scripts/run_mix100_cycle_mixer.py \
  --history-root /root/autodl-tmp/BatLiNet-main3 \
  --micro-batch-size 16 \
  > logs/mix100_cycle_mixer_full.log 2>&1 < /dev/null &
```

查看进度：`tail -f /root/autodl-tmp/BatLiNet-cycle-mixer/logs/mix100_cycle_mixer_full.log`。
只读核对：`python -B scripts/run_mix100_cycle_mixer.py --audit-only`。
本地检查：`python -B scripts/test_mix100_cycle_mixer.py`。
