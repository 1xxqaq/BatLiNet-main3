# MIX-20：软匹配后的潜在差分

本入口比较基础潜在交叉注意力与软匹配差分版，各八个随机种子从头训练。
代码实现与本地检查不代表已取得服务器正式实验结果。

## 模型

目标和参考保留原共享二维卷积编码器。每个注意力头计算：

- Q = Wq LNq(T)，K = Wk LNkv(R)。
- Vt = Wv LNkv(T)，Vr = Wv LNkv(R)：两侧使用同一个归一化层和值投影。
- A = softmax(Q K^T / sqrt(d))，D = Vt - A Vr。
- 各头差分拼接，经原输出投影、随机失活、原残差前馈网络，再进入原参考预测头。

训练保留注意力权重随机失活，此时随机失活后的权重不保证行和为1；评价关闭随机失活。
不保留原始目标词元的直接残差跳连，不新增上下文拼接、门控、双向注意力或逐参考损失。
仅允许一层匹配；重复对差分词元匹配原参考的语义没有纳入本实验。
两模型MIX-20参数量均170210；同种子初始参数及初始化后的随机数状态一致。
注意力实现路径不同，不声称训练中的随机失活掩码逐位一致。
多词元自比较不保证零差分，最终预测头也不施加自比较为零或反对称约束。

## 固定协议

207训练／147测试，无验证集。各1000轮，使用第1000轮，不按测试指标选择权重。
批量128、float32、不打乱、AdamW学习率0.001及权重衰减0.01，沿用legacy_train。
训练2个参考取均值；测试旧协议32个参考取PyTorch较小中位值。
标签为历史训练集自然对数及样本标准差标准化，两分支在此空间等权融合。
全部16次训练完整核对后统一测试。历史原始BatLiNet和历史潜在模型结果仅作单列参照。
当前基础模型重新训练，不依赖此前循环轴或双轴权重。

入口审计历史数据、标签、顺序及八份固定参考名单，不重新抽样生成测试协议。
每个种子保存run.json、train.jsonl、epoch1000.pt、test.pt；总表包含历史16行及新增16行。
保存配置、源码指纹、运行环境、实际参考索引、分支预测和原单位预测。
完整运行核对后跳过；不完整或来源不一致的目录报错，不覆盖、不自动续训。
同一工作区不要并发启动多个入口。权重不包含优化器续训状态。

## 服务器命令

在BatLiNet-cycle-mixer工作树、agent/cycle-mixer-latent-cross-attention分支执行：

```bash
conda activate batlinet
cd /root/autodl-tmp/BatLiNet-cycle-mixer
git pull --ff-only origin agent/cycle-mixer-latent-cross-attention
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
python -B scripts/test_soft_aligned_difference.py
python -u -B scripts/run_mix20_soft_aligned_difference_207.py --audit-only
mkdir -p logs
nohup python -u -B scripts/run_mix20_soft_aligned_difference_207.py \
  > logs/mix20_soft_aligned_difference_207_full.log 2>&1 < /dev/null &
tail -f logs/mix20_soft_aligned_difference_207_full.log
```

历史目录默认/root/autodl-tmp/BatLiNet-main3及/root/autodl-tmp/BatLiNet-main2。
可用--history-root、--original-root、--protocol-dir显式指定。
默认新结果目录workspaces/mix20_soft_aligned_difference_207_v1。
--audit-only只核对，不写入结果或开始训练；普通入口自动完成训练及测试，无需--run-test。

## 本地检查范围

测试覆盖显式公式对照、共享值空间、自比较单词元、参考置换、梯度、初始参数和随机数、
20／100循环前向形状、原损失及均值／中位数、16次训练完成后测试、重复复用与损坏拒绝。
流程测试使用小型合成数据，每次只真实训练一轮，其余完成记录仅模拟流程。
100循环仅检查模型形状，未提供本轮MIX-100正式训练入口。
