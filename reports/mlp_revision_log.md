# MLP 输入设计比较

## 初始方案：state-concat MLP

第一版把 2000 维同批次 control pseudobulk 与 perturbation multi-hot 直接拼接，
再用两层 MLP 预测表达变化。固定 seed 42，在 RTX 4060 上训练后得到：

| 拆分 | 最佳验证 MSE | 测试 MSE | 测试响应 Pearson |
| --- | ---: | ---: | ---: |
| seen | 0.008461 | 0.010288 | 0.4921 |
| unseen_combo | 0.012599 | 0.006705 | 0.5540 |
| unseen_gene | 0.013014 | 0.011915 | 0.4117 |

最终 MLP 使用下面的 perturbation-response 设计。选择依据是 state-concat 的验证 MSE
弱于 Ridge；上表同时保留当时的测试结果，记录输入设计的变化。

## 修改原因

在当前 pseudobulk 设计中，每个拆分只有少数不同的 batch-level control reference。
2,000 维 reference 相比 105 维扰动身份增加了大量输入维度。
批次噪声拟合是对初版验证表现较差的一种解释，实验没有单独验证这一机制。

## 修订方案：perturbation-response MLP

修订后的 MLP 只预测 perturbation delta：

```text
perturbation multi-hot -> MLP -> predicted delta
matched control + predicted delta -> predicted expression
```

control state 仍通过明确的残差连接参与最终预测。训练集中从未出现的基因会被 mask；
若没有任何已知 perturbation，模型输出 zero delta。
Mini-VC 则另行编码 control state，并在 latent space 中预测扰动更新。
