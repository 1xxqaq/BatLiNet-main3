# 原 latent 模型：当前环境下的 207 电池同条件重训

入口为 `scripts/run_mix20_latent_current_207.py`，训练名称为 `latent_cross_attention_current`。
实际模型类是原始 `LatentCrossAttentionBatLiNetRULPredictor`，直接读取其历史 MIX-20 配置，参数量为 170210；不包含循环轴残差模块。

新入口直接调用增强版实验的 `legacy_train`，保持 207 个训练电池、六通道输入及顺序、历史标签变换、批量 128、完整 float32 精度、AdamW、1000 轮、随机参考数量及每 100 轮的随机数状态处理一致。
在训练前先核对历史 16 份预测以及已有增强版八份完整训练、权重和测试结果，按相同数据和实际固定参考名单比较。
增强版结果仅加载及重算指标；新增计算是原模型八次训练及八次固定测试。

旧的 `run_mix20_cycle_mixer_207.py`、模型代码和配置均未修改，因此已有增强版实验的源码指纹校验保持有效。
新实验独立保存到 `workspaces/mix20_latent_current_207_v1`，不覆盖旧实验。
每个种子保存 `run.json`、`train.jsonl`、`epoch1000.pt` 和 `test.pt`，记录当前 Python、PyTorch、CUDA、cuDNN、显卡及浮点计算设置。
原增强版未保存这些完整环境信息；本次重训提供当前环境下的基线对照，不声称能还原历史环境的全部差异。

所有八个种子训练完整且通过核对后自动进行固定测试，最后生成 `summary.json`，包括历史 BatLiNet、历史 latent、本次 latent 重训及已有增强版共 32 行结果。
控制台还显示增强版相对本次原模型的平均优势及八种子胜率；MAPE、ACC15 均为小数比例。
每 20 轮显示本种子预计剩余时间；完整训练会核对后跳过，不支持从中断轮次继续。

## 服务器运行

先更新工作树到包含本入口的提交，再执行：

```bash
conda activate batlinet
cd /root/autodl-tmp/BatLiNet-cycle-mixer
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
mkdir -p logs
nohup python -u -B scripts/run_mix20_latent_current_207.py \
  --history-root /root/autodl-tmp/BatLiNet-main3 \
  --original-root /root/autodl-tmp/BatLiNet-main2 \
  --enhanced-runs /root/autodl-tmp/BatLiNet-cycle-mixer/workspaces/mix20_cycle_mixer_207_v1 \
  > logs/mix20_latent_current_207_full.log 2>&1 < /dev/null &
```

查看进度：

```bash
tail -f /root/autodl-tmp/BatLiNet-cycle-mixer/logs/mix20_latent_current_207_full.log
```

只核对已有材料而不训练或写文件：

```bash
python -B scripts/run_mix20_latent_current_207.py --audit-only
```

本地检查：

```bash
python -B scripts/test_latent_current_207.py
```
