# Mini-VC 训练与评价

本报告由 `scripts/summarize_mini_vc.py` 直接读取 `results/mini_vc/combined_model_comparison.csv` 生成。

| 拆分 | 模型 | 测试条件数 | 全基因 MSE↓ | 响应 Pearson↑ | Top-100 DE Pearson↑ | Top-100 恢复率↑ |
| --- | --- | --- | --- | --- | --- | --- |
| seen | mini_vc | 236 | 0.0035 | 0.7454 | 0.8639 | 0.5596 |
| seen | mlp | 236 | 0.0033 | 0.7681 | 0.8915 | 0.5731 |
| seen | ridge | 236 | 0.0036 | 0.7641 | 0.8870 | 0.5654 |
| unseen_combo | mini_vc | 13 | 0.0019 | 0.9142 | 0.9560 | 0.7431 |
| unseen_combo | mlp | 13 | 0.0017 | 0.9188 | 0.9659 | 0.7523 |
| unseen_combo | ridge | 13 | 0.0017 | 0.9006 | 0.9432 | 0.7485 |
| unseen_gene | mini_vc | 27 | 0.0113 | 0.4221 | 0.4739 | 0.4559 |
| unseen_gene | mlp | 27 | 0.0109 | 0.4584 | 0.5243 | 0.5007 |
| unseen_gene | ridge | 27 | 0.0095 | 0.5224 | 0.5785 | 0.4926 |

## 按拆分解读

- `seen`：Mini-VC MSE 比最佳基线 `mlp` 高 5.2%；响应 Pearson 最高模型为 `mlp`。
- `unseen_combo`：Mini-VC MSE 比最佳基线 `mlp` 高 10.0%；响应 Pearson 最高模型为 `mlp`。
- `unseen_gene`：Mini-VC MSE 比最佳基线 `ridge` 高 18.4%；响应 Pearson 最高模型为 `ridge`。

## Mini-VC checkpoint

| 拆分 | 最佳 epoch | 训练 epoch | 最佳 validation response MSE | 设备 |
| --- | --- | --- | --- | --- |
| seen | 280 | 320 | 0.0036 | cuda |
| unseen_combo | 213 | 253 | 0.0056 | cuda |
| unseen_gene | 19 | 59 | 0.0083 | cuda |

每个 checkpoint 均由 validation response MSE 选择；test 不参与 early stopping 或学习率调整。

## 结论边界

Mini-VC 学会了可用的 latent transition，但 V0.1 中并未稳定超越更简单的 Ridge/MLP。unseen-gene 尤其受 identity-only perturbation 表示限制。这说明增加 Encoder–Transition–Decoder 结构本身不保证更好的泛化。

三种拆分均保存 checkpoint、training log、逐条件指标和测试预测。两次完整训练的日志哈希一致；checkpoint 重载预测也通过数值核对。
