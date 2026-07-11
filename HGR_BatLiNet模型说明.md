# HGR-BatLiNet 模型说明

`HGRBatLiNetRULPredictor` 是在 `latent_cross_attention` 基础上重新设计的
层次化增益感知关系模型。现有 `latent_cross_attention` 代码和配置保持不变。

## 核心结构

1. **双轴层次化退化编码**
   - 先沿容量轴编码每个循环的局部曲线形态；
   - 再构造相对首循环变化和相邻循环变化；
   - 使用循环级 Transformer 建模早期退化演化。
2. **门控双向目标—参考关系**
   - 同时计算目标读取参考、参考读取目标的交叉注意力；
   - 通过零初始化的门控残差注入关系信息，避免直接覆盖单电池表示。
3. **参考纠偏收益学习**
   - 每个参考预测 `support_label + delta`；
   - 打分头学习该参考相对自身预测能够带来的纠偏收益；
   - 训练同时使用连续收益监督和成对排序监督。
4. **稳健集合聚合与自适应校正**
   - 将学习收益、预测不确定性和候选离群程度联合用于参考权重；
   - 用目标级校正门动态控制参考分支的注入强度；
   - 最终预测为 `y_ori + gate * aggregated_correction`，不再使用固定 `alpha`。

## 配置与代码

```text
src/models/rul_predictors/hgr_batlinet.py
configs/ablation/diff_branch/hgr_batlinet/mix_20.yaml
scripts/smoke_test_hgr_batlinet.py
```

正式配置训练时随机使用 4–8 个参考，测试时使用固定协议中的 32 个参考。
训练和测试使用相同的稳健聚合逻辑。

## 可配置消融

```yaml
use_bidirectional_relation: True
use_gain_aware_weights: True
use_adaptive_gate: True
```

依次关闭以上开关，可以分别评估双向关系、收益感知权重和自适应校正门。
`fixed_gate` 用于关闭自适应门后的固定融合消融。

## 诊断字段

固定协议预测结果除原有字段外，还会输出：

```text
y_ori_log_variance
support_delta
support_log_variance
support_score
support_weight
correction_gate
support_dispersion
support_entropy
final_prediction
```

这些字段可用于检查参考收益排序、权重集中程度、不确定性、集合分歧和动态门控行为。

## 无数据冒烟测试

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
python scripts/smoke_test_hgr_batlinet.py
```
