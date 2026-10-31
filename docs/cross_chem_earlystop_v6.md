# 跨化学体系迁移：每10次更新评估的逐次停止对照 v6

相对于v5，本版仅将评估间隔从50次参数更新改为10次。最大1000次更新、最早100次后允许停止、连续5次误差未严格下降则停止，以及保留首次达到最低误差权重的规则保持一致。不设额外改善幅度阈值，不执行第0步评估。

第10步可以成为最终最佳步数。若误差从首次评估起始终不改善，第60步虽已累计5次未改善，仍须等待第100步才允许停止；若最后最佳点为第50步则最早第100步停止，最后最佳点为第60步则最早第110步停止。触及1000步时停止原因仍记为`max_steps`，最终使用全过程最佳权重。

训练跑满1000次时，评估次数从v5的20次增加为100次，即5倍；评估开销增加，实际总耗时取决于是否提前停止，不能据此保证训练更快。

## 训练、监控与测试

继续复用v3实际保存的112份目标训练、测试名单、已选学习率，以及底层实际80%训练对和20%验证对。275块LFP全部保留，无32次开发实验，无40块LFP留出。数据和训练模块复用v5实现；全部8次来源预训练与112个目标场景×3种方法重新训练，共344次，完成并核验后统一计算336份测试预测。

| 阶段 | 每10次更新监控的误差 |
| --- | --- |
| LFP来源预训练 | 275块LFP训练电池的对数寿命预测均方误差 |
| 目标直接训练 | 已选目标训练电池的对数寿命预测均方误差 |
| 目标微调 | 已选目标训练电池的对数寿命预测均方误差 |
| 组合模型 | 留出20%电池对的关系差分对数均方误差 |

评估时关闭随机失活，不计算梯度，并隔离随机数和训练模式。单电池阶段属于训练误差停滞停止，组合验证共享目标训练电池，不能表述为独立目标电池验证。组合指标决定整个联合模型的最佳快照；训练损失及最终275参考融合方式沿用v5。微调分别从对应种子的最佳来源预训练权重开始，重新建立优化器。

v6使用独立代码入口与结果目录，不加载旧v5权重或进度，不将两种评估间隔的结果混合恢复。其余实验设定和解释边界见[v5说明](cross_chem_earlystop_v5.md)。

## 部署、运行与恢复

服务器首次使用独立工作树部署。将`<本次分支>`和`<本次提交>`替换为实际交付值：

```bash
cd /root/autodl-tmp/BatLiNet-cycle-mixer
git fetch origin <本次分支>
git worktree add --detach /root/autodl-tmp/BatLiNet-cross-chem-earlystop-v6 <本次提交>
cd /root/autodl-tmp/BatLiNet-cross-chem-earlystop-v6
bash scripts/start_cross_chem_earlystop_v6_nohup.sh \
  --previous-workspace /root/autodl-tmp/BatLiNet-cycle-mixer/workspaces/cross_chem_matched_v3
```

启动脚本内部使用`nohup`，自动进入`batlinet`环境，设置`OMP_NUM_THREADS=8`与`MKL_NUM_THREADS=8`。默认结果目录为新工作树内的`workspaces/cross_chem_earlystop_v6`。旧工作树、v5代码、配置和结果保持原样。

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
