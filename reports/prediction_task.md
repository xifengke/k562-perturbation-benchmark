# K562 扰动响应预测任务

## 任务定义

模型学习 K562 细胞群体在给定 CRISPRa 扰动条件下的平均转录响应。
reference 和 target 来自同批次的独立细胞：

```text
同一实验批次的 control/reference pseudobulk + perturbation identity
                                ↓
                         transition model
                                ↓
该批次、该 perturbation 的 expression pseudobulk / delta expression
```

Norman 数据的 237 个条件全部出现在 8 个 `gemgroup` 中。过滤
`good_coverage=False` 后，共有 1,896 个非空的 condition × gemgroup 组合；每个组合的
细胞数中位数为 42。它们可作为天然的、互不重叠的 pseudobulk replicates。

## 预测单位

一个建模样本由以下内容组成：

- `x_control,g`：gemgroup `g` 中 control cells 的 pseudobulk expression；
- `a_p`：perturbation `p` 的表示；
- `y_p,g`：同一 gemgroup 中 perturbation `p` cells 的 pseudobulk expression；
- `delta_p,g = y_p,g - x_control,g`：相对同批 control 的表达变化。

这里的 `x_control,g` 和 `y_p,g` 来自不同细胞。它们只在 condition/batch 层面对应，
不解释为真实时间轨迹。

## 输入与输出

### Expression input

- QC 后的 sparse UMI counts；
- 每个细胞按 library size 缩放至 10,000，再进行 `log1p`；
- 每个划分只用训练细胞选择 2,000 个 HVG；
- 模型输入为 library-size normalized、`log1p` 后的 pseudobulk 表达。

2,000 个基因包含 102 个可在矩阵中匹配的扰动目标。实验没有进行 3,000 HVG 的比较。

### Perturbation input

- control：全零 multi-hot；
- single：一个 gene ID 对应的 multi-hot 位为 1；
- double：两个 gene ID 对应的位为 1；
- Mini-VC 的双扰动使用两个 embedding 的和，表示与基因顺序无关。

数据中的 131 个双基因条件涉及 73 个基因，所有组成基因都有对应的单基因条件。因此
unseen-combination evaluation 可以严格做到“组合没见过，但两个组成基因分别见过”。

### Model output

模型输出长度为 `n_hvg` 的向量：

1. 主要输出：预测的 perturbed log-expression `y_hat`；
2. 派生输出：`delta_hat = y_hat - x_control`；
3. 不输出某个具体细胞的确定未来状态。

MLP 训练预测 delta，Mini-VC 联合优化 expression、重建和 latent alignment 损失。
评价使用条件质心的表达及 delta，不进行单细胞分布匹配。

## Evaluation splits

### A. Seen-perturbation

训练、验证、测试包含相同 perturbation，但使用互不重叠的细胞/pseudobulk replicate。

它回答：模型能否重建已经观察过的扰动响应？这是拟合能力和实现正确性的检查，不能作为
zero-shot 泛化结论。

为了避免同一 pseudobulk 被拆开，split 的最小单位是 pseudobulk replicate，不是矩阵元素。

### B. Unseen-gene perturbation stress test

测试集中保留训练时从未作为 perturbation 出现的单基因，并从训练集中移除所有包含这些基因的
双扰动。

identity-only MLP 和 Mini-VC 用训练基因 mask 屏蔽未知扰动身份。
MLP 在没有已知操作时输出 zero delta；Mini-VC 在同样情况下保持 `z' = z`，
输出为 reference 的自编码器重建，并不保证与 reference 逐元素相等。
Ridge 的未知基因列没有训练效应，预测仍可包含已知基因项和拟合的截距。
这些模型没有统一的 `<UNK_PERT>` token。scGPT 则使用多数基因的预训练 token，
并为三个词表外基因设置单独的可训练 fallback embedding。

### C. Unseen-combination（V0.1 主泛化任务）

测试集包含从未训练过的 `A+B`，但 A 和 B 的单扰动都在训练集中。训练、验证、测试按完整
perturbation condition 划分，测试 condition 的细胞不会进入 HVG fitting、模型训练或 early stopping。
library-size normalization 使用各细胞自身的 counts，不拟合跨细胞的归一化统计。

它回答：模型能否把已知单基因响应组合成未见过的双基因响应？这与 Norman 实验设计最匹配，
也是 V0.1 最可信的“unseen perturbation condition”任务。

## 防止 data leakage

实现中的划分和模型选择规则：

1. cell barcode 只属于一个 split；
2. unseen split 先分 condition，再进行所有数据驱动的拟合；
3. HVG mean/variance 和 dispersion 只由 train cells 估计；library-size normalization 逐细胞计算；
4. validation 用于 early stopping/模型选择；Ridge 选定 alpha 后在 train+validation 上重拟合；
5. test control cells 与 train control cells分开，用 test control 仅构造测试参考和真实 DE 对照；
6. DE ground truth 只从对应 split 的 observed perturbed/control cells 计算；
7. 同一个双扰动的不同 `gemgroup` 不可跨越 unseen-combination 的 train/test 边界。

## Baseline 对应关系

- No-change：`y_hat = x_control`。
- Mean-response：预测训练 perturbations 的平均 delta。
- Ridge：以 single/double multi-hot gene identity 预测 delta。
- MLP：multi-hot identity 预测 delta，再加 reference expression。
- Mini-VC：`Encoder(x_control) → Transition(z, perturbation) → Decoder(z')`。

所有模型使用相同 split、HVG、pseudobulk 与 evaluation code，不能为某个模型单独改变测试集。

## 评价指标

每个 perturbation 先单独计算，再进行 macro average：

- all-HVG MSE、RMSE、Pearson、Spearman；
- delta-expression Pearson、Spearman；
- true DE genes 上的 MSE/相关性；
- top-20、top-50、top-100 DE gene recovery；
- predicted/observed response vector 的 cosine similarity；
- 每个 split 和每个 perturbation 的 cell count、预测值与失败情况。

no-change 与其他基线使用相同的评价代码。DE 基因按对应 split 中真实 delta 的绝对幅度排序，
这里不进行差异表达显著性检验。报告同时保留整体分数和逐条件分数。

## 范围

- 不预测同一细胞的真实时间演化；
- 不把随机 control–perturbed cell 配对当作监督真值；
- 不处理 FASTQ/Cell Ranger；
- 基础模型不含 GRN 或 DNA sequence；额外实验分别测试 scGPT 迁移和静态 Reactome 特征；
- 不使用测试集选择 HVG、超参数或 checkpoint；测试 DE 排名仅作为评价真值。
