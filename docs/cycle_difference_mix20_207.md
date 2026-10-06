# MIX-20循环轴模型：原始输入与自身循环差分对照

本文件记录已实现的输入处理与运行入口。尚未执行服务器正式训练，没有新实验结果。未修改根目录交接资料或旧模型、旧实验入口。

## 代码与输入定义

- 模型：`src/models/rul_predictors/cycle_difference_batlinet.py`。
- 入口：`scripts/run_mix20_cycle_difference_207.py`。
- 检查：`scripts/test_cycle_difference_207.py`。
- 继承现有循环轴残差模型，仅替换输入准备；目标与参考仍独立经过同一共享编码器。
- `cycle_raw_current`：原始六通道曲线，使用现有清理流程。
- `cycle_self_difference`：先用相同流程清理原始曲线，再对每个电池独立计算 `X[:, :, h, :] - X[:, :, 0, :]`。
- 第1次循环为固定基准，差分后为零。不自动改换基准；某电池基准循环全零时停止并报错。
- 在减法前记录清理后全零循环，减法后重新置零；MIX-20索引10的第11次循环保持无效，不生成负基准曲线。其他全零循环同样处理。
- 不重复清理差分曲线，不做目标减参考的原始输入差分，不拼接原始／差分通道，不新增描述量、标准化或掩码注意力。
- 目标训练数据、随机训练参考及固定测试参考均走同一准备方法；已准备的参考不重复差分。
- 两版均为170556参数，编码器、两预测头、标签变换、随机数初始化流程一致。模型权重中的参数键也与原循环轴模型一致；运行配置额外记录输入模式。

## 实验协议

使用历史完整数据包及固定参考协议：207训练／147测试，无验证集。模型种子0—7，两版各8次从头训练，共16次；不加载旧训练权重。原始输入本轮重训用于直接配对，旧BatLiNet和基础潜在交叉注意力只作历史参照。

共用旧`legacy_train`函数：固定1000轮，批量128、梯度累积1、不打乱、float32，AdamW学习率0.001、权重衰减0.01，无新增裁剪。每100轮只模拟旧监测的随机抽样，不计算测试预测。16次训练全部完成并通过核对后，才统一用第1000轮权重测试。

训练每目标随机取2个参考并取均值；测试复用历史每目标32个固定参考，采用PyTorch较小中位值。标签取自然对数，用207个训练标签的历史均值及样本标准差标准化；两分支在此空间按0.5等权融合，逆变换为寿命循环数。

只读预检查核对历史16份预测的数据、标签变换、电池顺序、实际参考名单及指标，检查两版输入使用的差分基准。它会读取历史资料及测试特征，不创建训练目录，不训练，不产生新测试预测。

## 服务器运行

```bash
conda activate batlinet
cd /root/autodl-tmp/BatLiNet-cycle-mixer
git fetch origin
git switch agent/cycle-mixer-latent-cross-attention
git pull --ff-only origin agent/cycle-mixer-latent-cross-attention
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

python -u -B scripts/run_mix20_cycle_difference_207.py --audit-only
```

核对通过后启动正式训练：

```bash
mkdir -p logs
nohup python -u -B scripts/run_mix20_cycle_difference_207.py \
  --history-root /root/autodl-tmp/BatLiNet-main3 \
  --original-root /root/autodl-tmp/BatLiNet-main2 \
  --workspace workspaces/mix20_cycle_difference_207_v1 \
  --device cuda:0 \
  > logs/mix20_cycle_difference_207_full.log 2>&1 < /dev/null &
```

查看进度：

```bash
tail -n 50 logs/mix20_cycle_difference_207_full.log
```

默认查找上述两个历史仓库之一的`artifacts/fixed_test_support_indices/mix_20/protocol_v1/seed_0.pt`至`seed_7.pt`。协议不齐时停止，不重新抽取；可用`--protocol-dir`明确指定完整旧协议目录。入口不依赖已有循环轴训练目录，不需要`--run-test`。

## 输出与重复执行

工作区：`workspaces/mix20_cycle_difference_207_v1`。两种模型各种子目录保存：

- `run.json`：配置、输入处理、参数量、数据／代码／固定参考指纹、运行环境与完成状态。
- `train.jsonl`：逐轮损失及耗时。
- `epoch1000.pt`：最终权重及来源信息，不含优化器状态，不是中途续训断点。
- `test.pt`：标准化和原单位预测、真值、标签统计量、两分支及32个单参考预测、实际参考索引、分支四指标及权重指纹。

`summary.json`含历史16行和新增16行，共32行。终端汇总RMSE、MAE、MAPE、ACC15，以及自身／参考聚合分支RMSE的八种子均值、样本标准差和配对胜率。寿命误差单位为循环；MAPE和ACC15为小数比例，结果不是八模型集成。

重复执行时，先检查全部已有目录及测试结果，已完成且来源一致的训练／测试直接复用。来源不符、预测不合法或训练未完整完成时停止，不覆盖或自动续训。中断目录需要单独保存后使用新的工作区重新启动。

## 本地验证

```powershell
& 'D:\GongZuo\Anaconda\azb\envs\batlinet\python.exe' -B scripts/test_cycle_difference_207.py
```

七项检查覆盖原始版与旧模型的权重、随机数、预处理及预测一致性；自身基准差分及无效循环；非法基准拒绝；训练／测试参考的一次准备；16次训练完成后测试、结果复用和错误名单拒绝；只读核对与未完成目录拒绝；历史资料缺失时不启动。

流程测试使用小数据及每次1轮真实CPU训练，再模拟1000轮完成记录，只验证入口控制流程，不代表正式1000轮训练。另已使用本地历史资料完成207／147数据及八份固定协议的只读核对，未执行正式GPU训练。
