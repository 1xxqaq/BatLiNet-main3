# 跨化学体系迁移：来源开发早停对照 v4

本实验比较目标直接训练、LFP预训练后微调、循环轴残差与软匹配差分组合三种方法。它在LFP开发数据上分别确定训练预算，再对正式目标训练电池重新训练。已有v3固定预算结果单独保留；v4不覆盖v3结果、权重、名单或日志。

这里的“早停”发生在LFP开发阶段。正式跨化学训练使用开发阶段选定的固定预算，不将其表述为“每个目标模型都在独立目标验证集上早停”。

## 只读来源

- 默认上一版本目录：`workspaces/cross_chem_matched_v3`。
- 从其`queue.json`与`selection.json`读取既有队列身份和已选学习率，不重新搜索学习率。
- 复用其实际保存的`pilot/split_0.json`、`pilot/split_1.json`，每份为219块LFP来源、16块LFP训练候选、40块独立LFP验证。各数量使用保存的候选顺序前n块，不能另抽名单。
- 正式目标使用v3的112份`protocols/<体系>/train_<n>/seed_<s>.json`实际名单：LCO、NMC为1/2/4/8/16块，NCA为1/2/4/8块，每档8种子。
- 电池特征继续从v3队列身份中的`existing_workspace`所指向的v2缓存读取。该缓存包含275块LFP来源、37块LCO、22块NCA、69块NMC。
- 缺少上述来源、名单不一致或文件指纹变化时停止，不生成替代划分，不转用另一数据集。

v3的模型权重不用于v4初始化。v4的开发预训练和正式预训练均从头开始。

## LFP开发早停规则

开发训练数量扩展为1、2、4、8、16，每个方法、每个数量、每个划分独立训练并按同一规则早停。每个划分先执行1次来源预训练，再执行5个数量×3种方法，共16次；两个划分共32次开发训练。

| 参数 | 本版固定值 |
| --- | --- |
| 最大优化器更新次数 | 1000 |
| 首次验证 | 第50次更新后 |
| 后续验证频率 | 每50次更新 |
| 允许停止的最小更新次数 | 100 |
| 耐心值 | 连续5次验证未严格改善 |
| 改善判定 | 新验证误差严格小于此前最佳误差 |
| 最佳权重 | 验证误差最低时的权重；相等时保留先出现的权重 |
| 验证指标 | 各方法最终寿命预测在固定对数标签空间的均方误差 |

不执行第0步验证，因此最佳步数至少为50。第50步建立初始最佳记录，此后未改善才累计耐心计数；100步之前禁止停止，但允许最佳权重位于第50步。训练最多执行1000次更新，即使达到上限仍使用验证误差最低的权重。

来源预训练单独使用同一规则。每个开发划分的微调实验，使用该划分自身开发预训练的最佳权重；五个训练数量从该最佳权重分别初始化，不接续前一个数量的微调结果。

组合模型的验证指标依据最终融合预测计算，使用当前开发划分的全部219块来源参考。三种方法均使用同一份40块LFP验证电池，不根据目标测试成绩选权重、修改耐心值或排除种子。

这些参数属于预先约定的对照规则，不声称由尚未回传的实际曲线确定为最优。

## 从开发早停转换为正式预算

每个“方法×训练数量”的正式预算，取两个LFP开发划分最佳步数的中位数，并向上取整到50的倍数。来源预训练预算独立按同样方式计算。例如两个最佳步数为100和250，中位数为175，正式预算为200次更新。

保留两个划分各自的最佳步数及是否达到训练上限，不能仅凭两个划分将中位预算解释为稳定最优值。NCA只使用其正式协议具有的1/2/4/8档位。

正式实验执行8次完整275块LFP来源预训练，以及112份目标名单×3种方法，共344次训练。三个目标方法使用各自“方法×数量”预算重新训练；同种子的14份微调任务分别从该种子的完整LFP预训练权重开始。全部344次训练完整通过检查后，才统一计算336份正式目标测试预测。

## 与v3保持一致的部分

模型仍使用同一组合结构及其提取出的单电池编码器和自身回归头。标签保持自然对数`ln(T/1000)`，不新增经验标准化。优化器、学习率、权重衰减、逻辑批量、微批量、浮点精度以及组合关系电池对顺序沿用v3；正式关系训练使用全部已选目标×275块LFP来源，不沿用历史80%/20%电池对划分。

测试仍使用正式保存的固定目标测试名单；组合版参考顺序遵循原名单，使用全部275块LFP参考并在对数空间聚合。结果保留全部8个种子，汇总均值、样本标准差以及同种子配对差异。

## 启动、查看与恢复

服务器首次部署时，保留现有`BatLiNet-cycle-mixer`目录的当前提交及其他实验文件，在独立工作树运行v4。下面的`<本次分支>`与`<本次提交>`应替换为本次交付的实际分支与提交；不要在旧目录执行切换提交或拉取并更新当前文件的操作。

```bash
cd /root/autodl-tmp/BatLiNet-cycle-mixer
git fetch origin <本次分支>
git worktree add --detach /root/autodl-tmp/BatLiNet-cross-chem-earlystop-v4 <本次提交>
cd /root/autodl-tmp/BatLiNet-cross-chem-earlystop-v4
bash scripts/start_cross_chem_earlystop_nohup.sh \
  --previous-workspace /root/autodl-tmp/BatLiNet-cycle-mixer/workspaces/cross_chem_matched_v3
```

`git fetch`及新增工作树不会更改旧目录当前提交，因此不改变其他任务的恢复身份。新目录默认输出为新工作树自身的`workspaces/cross_chem_earlystop_v4`；v3数据与结果仍位于旧工作树。

脚本内部使用后台启动方式，无需在外层再加`nohup`。脚本自动进入`batlinet`环境，设置`OMP_NUM_THREADS=8`与`MKL_NUM_THREADS=8`。在新工作树可分别执行来源检查与准备、仅执行32次开发训练、只读报告：

```bash
bash scripts/start_cross_chem_earlystop_nohup.sh \
  --previous-workspace /root/autodl-tmp/BatLiNet-cycle-mixer/workspaces/cross_chem_matched_v3 --prepare-only
bash scripts/start_cross_chem_earlystop_nohup.sh \
  --previous-workspace /root/autodl-tmp/BatLiNet-cycle-mixer/workspaces/cross_chem_matched_v3 --pilot-only
bash scripts/start_cross_chem_earlystop_nohup.sh \
  --previous-workspace /root/autodl-tmp/BatLiNet-cycle-mixer/workspaces/cross_chem_matched_v3 --report
```

也可显式指定路径和设备：

```bash
bash scripts/start_cross_chem_earlystop_nohup.sh \
  --previous-workspace /root/autodl-tmp/BatLiNet-cycle-mixer/workspaces/cross_chem_matched_v3 \
  --workspace /root/autodl-tmp/BatLiNet-cross-chem-earlystop-v4/workspaces/cross_chem_earlystop_v4 \
  --device cuda:0
```

默认新目录为`workspaces/cross_chem_earlystop_v4`，进程号保存在`queue.pid`，最近一次日志路径保存在`latest_log.txt`。

```bash
tail -f "$(cat workspaces/cross_chem_earlystop_v4/latest_log.txt)"
```

重复启动会检查进程号和队列锁。启动器对v3、实际来源缓存目录以及已知迁移和MIX实验目录的既有锁文件，只读打开并检查共享锁；同时检查新工作树与v3所在的旧工作树的工作区，避免在独立工作树部署后漏掉旧目录中的训练。若另一个训练队列持有排他锁则明确拒绝，不停止该进程，也不创建、截断或清理旧锁文件。该检查较为保守：即使通过`--device`指定另一设备，已知训练队列仍在运行时也会拒绝。

中断后进入已创建的新工作树，重新运行同一启动命令恢复，无需再次创建工作树：

```bash
cd /root/autodl-tmp/BatLiNet-cross-chem-earlystop-v4
bash scripts/start_cross_chem_earlystop_nohup.sh \
  --previous-workspace /root/autodl-tmp/BatLiNet-cycle-mixer/workspaces/cross_chem_matched_v3
```

已保存的来源指纹、开发划分、预算、权重和恢复状态需与本版本绑定一致。不要通过删除旧队列结果、修改v3选择文件或改写已保存名单来绕过检查。
