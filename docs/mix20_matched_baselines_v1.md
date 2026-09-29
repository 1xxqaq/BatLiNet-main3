# MIX-20 同条件基线对照入口 v1

本入口新增文件，不修改原始 `batlinet.py`、`latent_cross_attention_batlinet.py`、原配置及已完成的新模型训练入口。适配器调用原类的输入清理与预测方法，不调用旧的 `fit`，因其训练过程中评价测试集。

## 数据与评价边界

- 以现有 `artifacts/context_grid_v1/mix20.pt` 的电池编号、顺序、标签为唯一划分依据。
- 从原处理数据逐电池提取原版六通道 `6×20×1000` 特征；不将新模型的三通道输入直接用于基线。
- 核对源文件 SHA256、原 MIX20 划分归属和重新计算的寿命标签；确认原划分中没有遗漏标签有效的电池。已有缓存只校验对应的上下文数据与电池／标签一致性，不自动覆盖。
- 原始 BatLiNet 保留循环差分基准 0、跨电池原始特征差分及差分清理；潜在交叉注意力保留原始单电池特征清理。两个模型均按原配置关闭循环异常过滤。
- 保留旧输入政策：不新增特征标准化。标签使用与新模型一致的训练集对数均值和总体标准差，不能当作历史完整训练的逐位复现。
- 固定 166 训练／41 验证／147 测试；参数更新和参考池均只用 166 个训练电池。训练期间只评价验证集。
- 每种子验证参考抽样种子为 `seed+100000`，与已完成的新模型训练一致。测试为 `seed+200000`。每个查询抽样 32 个参考，允许重复；协议记录实际索引及双方编号与顺序。
- 各模型同一评价种子共享参考名单。模型初始化消耗随机数不同，所以同一训练种子不保证训练时每次抽样完全相同；本实验仍包括训练与参考抽样随机性。

## 对齐训练规则

两个基线各跑种子 0～7；1000 轮，每 25 轮验证，融合 RMSE 严格降低时保存 `best.pt`。AdamW 学习率 0.001、权重衰减 0.01；批量 8、累积 16、尾批按实际样本数加权；每轮打乱、bfloat16 混合精度、梯度裁剪 5。训练参考 2 个取均值；验证 32 个取 PyTorch 中位数（偶数时取较小中位值）。标签空间等权分支损失与等权融合，最后逆变换到寿命。评价使用全精度，分块仅用于节省显存，最终对完整 32 个结果取一次中位数。

这是统一选择规则下的完整方案对照，不是仅替换编码器的消融，也不是复现旧的 207 电池训练流程。

## 服务器目录

所有命令在 `/root/autodl-tmp/BatLiNet-context-grid-v1` 执行：

```text
artifacts/context_grid_v1/mix20.pt                原新模型缓存，不覆盖
artifacts/mix20_matched_baselines_v1/mix20.pt      新增旧格式特征缓存，约 163 MiB
protocols/mix20_matched_v1/{val,test}_seed*.pt     两种评价协议
workspaces/context_grid_v1/                       已完成新模型结果，不覆盖
workspaces/mix20_matched_baselines_v1/
  batlinet_seed0/ … batlinet_seed7/
  latent_cross_attention_seed0/ … latent_cross_attention_seed7/
evaluations/mix20_matched_v1/                      明确运行最终测试时才产生预测
```

每个基线运行目录保存 `run.json`、`train.jsonl`、`best.pt`。完成且参数／数据一致的任务可以跳过；不完整目录停止并报告，不支持自动续训或覆盖。准备特征只读取现有数据，不下载外部数据。

## 同步和训练

现有服务器目录是分离 HEAD 的 Git 工作树，可以正常 fetch 后切换到同一分支的新提交，不需要重新克隆。不使用 reset --hard 或 clean，不移动数据与结果。确保该目录无训练运行、无自行修改的跟踪代码，然后：

```bash
cd /root/autodl-tmp/BatLiNet-context-grid-v1
git status --short
git fetch origin agent/context-grid-lifetime-v1
git switch --detach origin/agent/context-grid-lifetime-v1
conda activate batlinet
bash scripts/run_mix20_matched_baselines.sh /root/autodl-tmp/BatLiNet-main2/data/processed
```

脚本先运行六项 CPU 测试，随后提取旧格式特征、保存协议、顺序训练两个基线各八个种子。单元测试中的轮次和电池数量是临时合成样例，与正式数据无关。

## 汇总与最终测试

全部完成后先审计 32 次训练和输出验证汇总，不计算测试指标：

```bash
python -B scripts/evaluate_matched_suite.py
```

检查要求四种模型都有完整 1000 轮日志、相同验证频率、匹配数据及关键配置、正确最佳权重。MAPE 和 ACC15 输出为小数比例。显式加入下列参数才评价测试集：

```bash
python -B scripts/evaluate_matched_suite.py --run-test
```

测试前先检查所有运行，再逐模型逐种子评价；已有预测只有来源指纹一致才复用。保存逐电池预测、原单位分支结果、参考索引、电池编号、数据／权重／协议 SHA256 及汇总 JSON。协议固定后不根据测试结果调整轮次、融合或选择种子。

## 本地检查范围

CPU 测试覆盖两种适配器与原类预测路径一致、输入缓存不被修改、训练梯度、参考分块与全量结果一致、协议错配拒绝、源文件与标签校验、两轮训练与权重加载、训练不读取测试特征、完成任务跳过及最终测试预检查。真实服务器数据和 CUDA 混合精度完整训练由服务器命令执行，不能用 CPU 合成测试代替其结果。
