# 循环轴残差增强版：历史 207 电池协议

本实验固定现有增强版结构及隐藏宽度 16，只检验它在历史 MIX-20 协议下的表现。
训练八个种子 0–7，每个种子固定训练 1000 轮，最后自动计算旧固定协议测试结果。
原始 BatLiNet 和 `latent_cross_attention` 复用历史八种子正式预测，不重新训练。

## 输入与固定评估

直接读取历史正式预测文件中的完整 `DataBundle`，复用 207 个训练电池、147 个测试电池的顺序、六通道特征和已拟合标签变换。
不从当前文件系统重新提取特征，不把 166/41 缓存直接拼接为新的参考索引顺序。
标签沿用自然对数及历史训练标签的样本标准差，区别于 166/41 入口使用的总体标准差。

启动时逐一核对两个旧模型的 16 份正式预测：完整特征及标签指纹、电池编号及顺序、标签变换统计量、指标重算，以及实际参考名单与旧协议的一致性。
旧协议要求 207 个训练参考池、147 个测试电池、每目标固定 32 个参考。
缺少历史结果或旧协议时直接报错，不重新生成参考名单。

历史目录默认是：

```text
/root/autodl-tmp/BatLiNet-main3/workspaces/fair_eval/batlinet_latent_cross_attention/mix_20/protocol_v1_formal
/root/autodl-tmp/BatLiNet-main2/workspaces/fair_eval/batlinet_original/mix_20/protocol_v1_formal
```

旧参考名单默认从以上两个仓库中的
`artifacts/fixed_test_support_indices/mix_20/protocol_v1/seed_0.pt` 至 `seed_7.pt` 寻找。
也可用 `--protocol-dir` 明确指定已有目录。

## 训练与检查点

- 使用历史 `latent_cross_attention` 配置及训练循环：每批 128，不打乱顺序，完整 float32 精度；梯度累积 1，AdamW 学习率 0.001、权重衰减 0.01，无额外梯度裁剪。
- 每目标随机抽取两个训练参考，损失及预测头沿用原模型。
- 原训练循环每 100 轮随机抽取参考并查看测试误差；本入口不计算这些测试预测或误差，只模拟对应的数据加载器和随机参考抽样对随机数状态的影响。
- 固定使用第 1000 轮，所有八种子训练完整并通过核对后，才计算增强版的固定测试预测。
- 原训练循环与本入口在小数据 CPU 检查中最终权重及随机数状态完全一致；不保证跨设备、软件版本的训练结果逐位一致。
- 每 20 轮显示训练损失、本种子预计剩余时间及后续种子数。时间根据当前平均速度估算，不是运行时限。
- 完整训练经核对后跳过；中断的训练目录会报错，不覆盖，也不支持中途续训。

三个模型最后按相同 32 参考取 PyTorch 中位数，两个分支在标准化标签空间等权融合，再逆变换到寿命单位。
控制台 MAPE、ACC15 均为小数比例，0.55 的 ACC15 即 55%。

## 服务器运行

在已更新到本次提交的 `BatLiNet-cycle-mixer` 工作树执行：

```bash
conda activate batlinet
cd /root/autodl-tmp/BatLiNet-cycle-mixer
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p logs
nohup python -u -B scripts/run_mix20_cycle_mixer_207.py \
  --history-root /root/autodl-tmp/BatLiNet-main3 \
  --original-root /root/autodl-tmp/BatLiNet-main2 \
  > logs/mix20_cycle_mixer_207_full.log 2>&1 < /dev/null &
```

只核对历史材料、显示两个旧模型指标而不训练或写文件：

```bash
python -B scripts/run_mix20_cycle_mixer_207.py --audit-only
```

查看日志：

```bash
tail -f /root/autodl-tmp/BatLiNet-cycle-mixer/logs/mix20_cycle_mixer_207_full.log
```

新增结果全部保存在 `workspaces/mix20_cycle_mixer_207_v1`。
每个种子有 `run.json`、`train.jsonl`、`epoch1000.pt`、`test.pt`；总表为 `summary.json`。
测试文件含标准化标签空间及原始寿命单位的预测、真实寿命、标签变换统计量、分支诊断、实际参考索引及历史数据和检查点指纹。
历史目录和 166/41 实验结果只读。

## 本地核对

```bash
python -B scripts/test_cycle_mixer_207.py
```
