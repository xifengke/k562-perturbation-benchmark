# 基础模型比较

本报告由 `scripts/summarize_baselines.py` 直接读取 `results/baselines/model_comparison.csv` 生成。

| 拆分 | 模型 | 测试条件数 | 全基因 MSE↓ | 响应 Pearson↑ | Top-100 DE Pearson↑ | Top-100 恢复率↑ |
| --- | --- | --- | --- | --- | --- | --- |
| seen | mean_response | 236 | 0.0086 | 0.4113 | 0.5422 | 0.3750 |
| seen | mlp | 236 | 0.0033 | 0.7681 | 0.8915 | 0.5731 |
| seen | no_change | 236 | 0.0105 | — | — | — |
| seen | ridge | 236 | 0.0036 | 0.7641 | 0.8870 | 0.5654 |
| unseen_combo | mean_response | 13 | 0.0065 | 0.5746 | 0.6589 | 0.4992 |
| unseen_combo | mlp | 13 | 0.0017 | 0.9188 | 0.9659 | 0.7523 |
| unseen_combo | no_change | 13 | 0.0094 | — | — | — |
| unseen_combo | ridge | 13 | 0.0017 | 0.9006 | 0.9432 | 0.7485 |
| unseen_gene | mean_response | 27 | 0.0108 | 0.4676 | 0.5119 | 0.4519 |
| unseen_gene | mlp | 27 | 0.0109 | 0.4584 | 0.5243 | 0.5007 |
| unseen_gene | no_change | 27 | 0.0136 | — | — | — |
| unseen_gene | ridge | 27 | 0.0095 | 0.5224 | 0.5785 | 0.4926 |

## 结果解读

- `seen`：MLP 相对 no-change 将全基因 MSE 降低 68.0%，MSE 比 ridge 低 5.9%；该拆分最低 MSE 模型是 `mlp`。
- `unseen_combo`：MLP 相对 no-change 将全基因 MSE 降低 81.9%，MSE 比 ridge 低 0.6%；该拆分最低 MSE 模型是 `mlp`。
- `unseen_gene`：MLP 相对 no-change 将全基因 MSE 降低 19.9%，MSE 比 ridge 高 14.3%；该拆分最低 MSE 模型是 `ridge`。
- no-change 的预测响应恒为零，因此响应相关性和 DE 排名没有数学定义，报告为 `—`，而不是让排序函数任意打破并列。
- `unseen_combo` 检验已见单基因效应能否组合到未见双扰动；加性 ridge 正好是这个假设的透明基线。
- MLP 使用 perturbation multi-hot 预测 delta，并通过 `predicted expression = matched control + delta` 形成最终输出。模型只用 validation early stopping；测试集不参与训练。
- `unseen_gene` 中测试基因从训练条件完全移除。当前 identity-only 多热表示没有这些基因的先验信息，因此该拆分主要量化这种表示的泛化上限。

## 数据流与关键代码

1. `src/mini_vc/data/dataset.py`：读取 pseudobulk，按 split 和 gemgroup 找到对应 control，并计算 `delta = perturbed - control`。
2. `src/mini_vc/models/baselines.py`：总体平均响应，以及带不惩罚截距的多输出 ridge 闭式解。
3. `src/mini_vc/models/mlp.py`：multi-hot 扰动响应网络；未知训练基因被显式 mask，control/完全未知操作输出 zero delta。
4. `src/mini_vc/training/torch_runner.py`：固定 seed、GPU/CPU 自动选择、early stopping、checkpoint 和训练日志。
5. `src/mini_vc/training/baseline_runner.py`：只用 validation 选择 alpha，随后用 train+validation 重拟合，最终只在 test 报告指标。
6. `src/mini_vc/evaluation/metrics.py`：先求每个 perturbation 的测试质心，再按条件宏平均，避免细胞数多的条件支配分数。

## 训练产物

no-change、mean-response、ridge 和 MLP 均在三种拆分上完成训练和评价。MLP checkpoint、逐 epoch 日志、测试预测和逐条件指标均已保存。初始 state-concat MLP 的结果及输入修改记录在 `reports/mlp_revision_log.md`。
