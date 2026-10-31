# 跨化学体系迁移：每10次更新评估的逐次停止对照 v6

相对于v5，本版仅将评估间隔从50次参数更新改为10次。最大1000次更新、最早100次后允许停止、连续5次误差未严格下降则停止，以及保留首次达到最低误差权重的规则保持一致。不设额外改善幅度阈值，不执行第0步评估。

第10步可以成为最终最佳步数。若误差从首次评估起始终不改善，第60步虽已累计5次未改善，仍须等待第100步才允许停止；若最后最佳点为第50步则最早第100步停止，最后最佳点为第60步则最早第110步停止。触及1000步时停止原因仍记为`max_steps`，最终使用全过程最佳权重。

训练跑满1000次时，评估次数从v5的20次增加为100次，即5倍；评估开销增加，实际总耗时取决于是否提前停止，不能据此保证训练更快。

## 训练、监控与测试

支持两种只读来源：有v3材料时，复用其已保存的实际名单与已选学习率；本机只有v2时，直接复用v2缓存、112份实际目标训练／测试名单和80%训练对／20%验证对，并显式固定学习率。后者不要求另一台服务器上的v3，不重跑开发，也不伪造v3选择文件。275块LFP全部保留，无32次开发实验，无40块LFP留出。数据审计和训练引擎复用v5实现；全部8次来源预训练与112个目标场景×3种方法重新训练，共344次，完成并核验后统一计算336份测试预测。

| 阶段 | 每10次更新监控的误差 |
| --- | --- |
| LFP来源预训练 | 275块LFP训练电池的对数寿命预测均方误差 |
| 目标直接训练 | 已选目标训练电池的对数寿命预测均方误差 |
| 目标微调 | 已选目标训练电池的对数寿命预测均方误差 |
| 组合模型 | 留出20%电池对的关系差分对数均方误差 |

评估时关闭随机失活，不计算梯度，并隔离随机数和训练模式。单电池阶段属于训练误差停滞停止，组合验证共享目标训练电池，不能表述为独立目标电池验证。组合指标决定整个联合模型的最佳快照；训练损失及最终275参考融合方式沿用v5。微调分别从对应种子的最佳来源预训练权重开始，重新建立优化器。

v6使用独立代码入口与结果目录，不加载旧v5权重或进度，不将两种评估间隔的结果混合恢复。其余实验设定和解释边界见[v5说明](cross_chem_earlystop_v5.md)。

## 当前服务器只有v2时

已确认当前服务器有`cross_chem_two_models_v2`，但没有`cross_chem_matched_v3`。使用新增的`--existing-workspace`模式直接导入v2；不需要先执行v3。导入仍严格核验实际缓存、403块电池、112份协议及文件指纹，缺失或变化时停止，不重新抽取名单。

本轮保留当前服务器v2实际保存的划分，三种方法共享同一本机名单。由于缺少另一台服务器的v3收据，不声称已经核验两台服务器之间的名单完全一致。

本轮已明确选择固定学习率`0.001`。这沿用旧v2的固定值，但没有证据证明它是另一台服务器上v3实际选中的值；记录中将明确标注为本轮显式固定学习率。`--learning-rate`在该模式下必须显式给出，三种方法及来源预训练共用此值。`--previous-workspace`与`--existing-workspace`互斥；v3模式不允许覆盖其已选学习率。

当前v6工作树已经创建，且此前在来源检查处退出、没有开始训练，因此更新此工作树即可，不要再次执行`git worktree add`。将`<本次提交>`替换为本次交付值：

```bash
cd /root/autodl-tmp/BatLiNet-cross-chem-earlystop-v6
git fetch origin agent/cycle-mixer-latent-cross-attention
git checkout --detach <本次提交>
bash scripts/start_cross_chem_earlystop_v6_nohup.sh \
  --existing-workspace /root/autodl-tmp/BatLiNet-cycle-mixer/workspaces/cross_chem_two_models_v2 \
  --learning-rate 0.001
```

该模式默认输出为`workspaces/cross_chem_earlystop_v6_v2`，与依赖v3的旧v6结果目录分开。执行前仍检查其他已知队列的锁，不停止旧进程。

```bash
tail -f "$(cat workspaces/cross_chem_earlystop_v6_v2/latest_log.txt)"
bash scripts/start_cross_chem_earlystop_v6_nohup.sh \
  --workspace workspaces/cross_chem_earlystop_v6_v2 --report
```

报告从指定结果目录读取保存的来源身份，不再预先检查默认v3目录。中断后用相同的`--existing-workspace`和`--learning-rate 0.001`启动命令恢复。两种来源模式、学习率或已保存名单不允许混合恢复。

## 部署、运行与恢复

如果服务器有完整v3材料，首次可按原方式使用独立工作树部署。将`<本次分支>`和`<本次提交>`替换为实际交付值：

```bash
cd /root/autodl-tmp/BatLiNet-cycle-mixer
git fetch origin <本次分支>
git worktree add --detach /root/autodl-tmp/BatLiNet-cross-chem-earlystop-v6 <本次提交>
cd /root/autodl-tmp/BatLiNet-cross-chem-earlystop-v6
bash scripts/start_cross_chem_earlystop_v6_nohup.sh \
  --previous-workspace /root/autodl-tmp/BatLiNet-cycle-mixer/workspaces/cross_chem_matched_v3
```

启动脚本内部使用`nohup`，自动进入`batlinet`环境，设置`OMP_NUM_THREADS=8`与`MKL_NUM_THREADS=8`。v3模式默认结果目录为新工作树内的`workspaces/cross_chem_earlystop_v6`；v2模式为`workspaces/cross_chem_earlystop_v6_v2`。旧工作树、v5代码、配置和结果保持原样。

```bash
tail -f "$(cat workspaces/cross_chem_earlystop_v6/latest_log.txt)"
```

只做来源检查和队列准备，或查看只读报告：

```bash
bash scripts/start_cross_chem_earlystop_v6_nohup.sh \
  --previous-workspace /root/autodl-tmp/BatLiNet-cycle-mixer/workspaces/cross_chem_matched_v3 --prepare-only
bash scripts/start_cross_chem_earlystop_v6_nohup.sh \
  --previous-workspace /root/autodl-tmp/BatLiNet-cycle-mixer/workspaces/cross_chem_matched_v3 --report
```

可通过`--workspace`指定独立结果目录，或通过`--device cuda:0`指定设备。没有开发阶段，不提供`--pilot-only`。

启动器只读检查已知旧队列和兄弟工作树的锁，包括v4、v5和其他v6队列。旧v5或其他已知训练仍在运行时明确拒绝，不停止旧进程，不修改旧锁。中断后进入已创建的v6工作树，重复同一启动命令恢复，无需再次创建工作树。恢复要求当前代码、环境、实际名单、电池对顺序与停止设置一致。
