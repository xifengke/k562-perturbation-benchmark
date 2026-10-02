# Mini-VC 模型结构

## 目标

Mini-VC 将同批次 control 表达和基因扰动身份映射到扰动后的平均表达。
reference 和 target 来自独立细胞，模型学习的是 pseudobulk 层面的映射。

## 数据流

```text
control pseudobulk x_ref
        ↓ Encoder
latent cell state z

perturbation multi-hot
        ↓ summed learnable gene embeddings
action embedding a

[z, a]
   ↓ residual Transition Network
z' = z + Δz
   ↓ Decoder
predicted perturbed expression
```

双扰动使用两个 gene embedding 的和，因此表示与基因顺序无关。训练期间未出现的基因
会被 mask；如果操作中没有任何已知基因，模型保持 `z' = z`。

## 训练目标

训练联合使用三个目标，权重依次为 1.0、0.2 和 0.1：

1. response MSE：预测扰动表达与真实扰动 pseudobulk 的差异；
2. reconstruction MSE：重建 reference 和 target，使 latent space 保留表达信息；
3. latent alignment：让 `z'` 接近 target 编码得到的 latent state，target latent 在该损失中停止梯度。

target expression 只在训练辅助目标中使用；推理只需要 reference 和 perturbation。

## 模型范围

- 输入输出是 2000 个 HVG 的 pseudobulk，而不是单细胞分布生成模型；
- perturbation 只有 identity embedding，没有通路、GRN 或 DNA sequence；
- summed embedding 是简单、可解释的双扰动组合假设；
- unseen-gene 没有可学习身份信息，只能退化为已知部分或 no-transition。

没有已知操作时，`z' = z`，但 Decoder 输出的 reference 重建仍可能存在误差。

## 前向与反向检查

`scripts/smoke_test_mini_vc.py` 使用真实 seen-split 训练 pseudobulk，在 RTX 4060 上
对 128 个样本执行了 20 次 forward/backward：

- 模型参数量：2,538,256；
- 输入/输出基因数：2000；
- latent dimension：128；
- 联合损失由 0.262823 降到 0.009512；
- 最终 response MSE：0.008270。

这是前向、反向和 GPU 数据流检查的训练损失，不用于比较测试性能。
正式训练的 early stopping、checkpoint 和三个划分的结果见 [训练报告](stage8_training_summary.md)。
