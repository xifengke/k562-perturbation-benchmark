# K562 Perturbation Benchmark

使用 Norman et al. (2019) 的 K562 CRISPRa Perturb-seq 数据，预测单基因和双基因扰动后的
平均转录组响应。输入为同批次的 control expression 和扰动身份，输出为 2,000 个基因的表达：

> 给定 K562 的 control/reference 转录组状态和一个单基因或双基因扰动身份，预测该扰动条件下的平均转录组响应。

项目已完成，包括基础模型比较、全部 131 个双基因组合的五折测试、scGPT 编码器迁移和
Reactome 通路特征对照。最初的 13 组合测试结果也保留在仓库中。

五折测试中，Ridge 的响应 Pearson 最高（0.8963），Mini-VC 的全基因 MSE 最低（0.001935），
MLP 的 Top-100 recovery 最高（0.7224）。scGPT 迁移和静态 Reactome 特征没有带来稳定提升。
完整数字见下方结果表和 [五折报告](reports/scgpt_fivefold_results.md)。

原因分析见 [Mini-VC 与 scGPT 性能讨论](reports/model_performance_analysis.md)。
Mini-VC 的整体误差更低，但强响应基因和排序指标较弱；scGPT 的微调范围仅涉及 control 编码器最后两层。
讨论结合训练规模、损失目标与保存预测，分别列出观测证据和未经过消融验证的解释。

![五折模型比较](results/figures/scgpt_fivefold_comparison.png)

## 1. 预测范围

模型学习 control 状态到扰动后平均表达的映射：

```text
reference state x + perturbation a
                 ↓
        learned transition F
                 ↓
predicted perturbed state y_hat
```

Mini-VC 用 `z' = F(z, a)` 表示 latent transition。输入输出都是转录组平均表达，
没有时间、空间或蛋白测量。Perturb-seq 没有同一个细胞扰动前后的配对测量，
因此 reference 和 target 使用同批次的独立细胞构建 condition-level pseudobulk。

## 2. 数据

| 项目 | 内容 |
|---|---|
| 原始研究 | Norman TM et al., *Science* (2019) |
| 论文 | *Exploring genetic interaction manifolds constructed from rich single-cell phenotypes* |
| DOI | `10.1126/science.aax4438` |
| GEO | `GSE133344` |
| 本项目使用版本 | scPerturb 1.4 整理的 `NormanWeissman2019_filtered.h5ad` |
| Zenodo | `10.5281/zenodo.13350497` |
| 文件大小 | 698,680,199 bytes（约 666.3 MiB） |
| 原始矩阵 | 111,445 cells × 33,694 genes |
| 扰动 | control、105 个单基因条件、131 个双基因条件 |
| 细胞类型 | K562 |
| 扰动方式 | CRISPR activation（CRISPRa） |

下载器会核对精确文件大小和 MD5，完整来源、下载链接、许可边界与引用方式见
[`data/README.md`](data/README.md)。详细数据审计见
[`reports/data_summary.md`](reports/data_summary.md)。

### 三种数据单位

1. 原始 AnnData 的一行是一个细胞，一列是一个基因，元素是 UMI count。
2. 建模样本是一个 `(perturbation, gemgroup)` pseudobulk，即同一条件、同一实验批次中
   QC 合格细胞的平均 log-normalized expression。
3. 评价时先把测试 pseudobulk 汇总为 condition centroid，再逐条件计算指标并做宏平均，
   避免细胞多的扰动主导分数。

QC 后保留 91,130 个细胞。每种 split 都只用训练细胞拟合 HVG，最终输入 2,000 个基因。
预处理细节见 [`reports/preprocessing_summary.md`](reports/preprocessing_summary.md)。

## 3. 精确预测任务

每个样本包含：

- `reference`：相同 split、相同 `gemgroup` 的 control pseudobulk，形状为 `[2000]`；
- `perturbation`：单/双基因的 multi-hot identity；
- `target`：对应扰动 pseudobulk expression，形状为 `[2000]`；
- `delta = target - reference`：相对同批次 control 的响应。

模型输出 `predicted_expression`，并由此得到
`predicted_delta = predicted_expression - reference`。target 只用于训练监督，推理时不是输入。

更完整的任务定义和防泄漏约束见
[`reports/prediction_task.md`](reports/prediction_task.md)。

## 4. Evaluation splits

| Split | 测试内容 | 能回答的问题 | 不能声称的结论 |
|---|---|---|---|
| `seen` | 已见扰动、未见 pseudobulk replicate | 模型能否拟合已观察的响应 | 不是 zero-shot 泛化 |
| `unseen_combo` | 未见双基因组合，但两个单基因分别见过 | 能否组合已知的单基因效应 | 不代表能泛化到全新基因 |
| `unseen_gene` | 训练中完全移除的基因及包含它的组合 | identity-only 表示的能力边界 | 当前版本不具备真正新基因语义 |

`unseen_combo` 是 V0.1 的主要泛化任务。所有 unseen split 都先按完整 condition 拆分；测试
condition 的细胞不会进入训练、HVG 选择、early stopping 或超参数选择。

V0.2 新增 `unseen_combo_fold0` 至 `unseen_combo_fold4`。每折测试 26 或 27 个完整双基因
条件，验证 13 个；五折合计 131 个，且每个双基因组合恰好测试一次。每折训练集始终保留
构成测试组合的单基因条件。审计结果见
[`reports/scgpt_fold_audit.json`](reports/scgpt_fold_audit.json)。这些条件属于同一个已被分析过的
数据集，所以五折覆盖也不等于独立外部验证。

## 5. 模型

### Baselines

- **No-change**：直接输出 reference。
- **Mean-response**：输出训练集平均 delta。
- **Ridge**：从扰动 multi-hot 线性预测 delta。
- **MLP**：从扰动 identity 非线性预测 delta，再通过 `reference + delta` 得到表达。

### Mini-VC

```text
reference expression ── Encoder ──> latent state z ───────────────┐
                                                                  ├─> residual Transition ─> z' ─> Decoder ─> y_hat
perturbation multi-hot ── summed gene embeddings ──> action a ────┘
```

双扰动使用两个 learnable gene embeddings 的和，因此不依赖基因顺序。Transition 预测 latent
residual；训练期还将真实 target 编码为 `z_target`，用于 latent alignment。损失为：

```text
L = 1.0 × response MSE
  + 0.2 × reconstruction MSE
  + 0.1 × latent alignment MSE
```

模型结构和 smoke test 见
[`reports/mini_vc_architecture.md`](reports/mini_vc_architecture.md)。

### V0.2：scGPT 迁移学习

使用[官方 scGPT](https://github.com/bowang-lab/scGPT) 的 whole-human 预训练 checkpoint。
编码器读取每个 split 与 `gemgroup` 中 4 个真实 control cells，每个细胞选用该折训练集确定的
512 个基因并按 checkpoint 的 51 档表达值输入。基因扰动的预训练 token embedding 经可训练
adapter，与 control 编码和原有 reference pseudobulk 一起送入新建的响应预测头，输出 2,000 个
目标基因的表达变化。target 来自另外的 perturbed cells，不构造单细胞前后配对。

- **冻结组**：固定 scGPT 编码器，只训练扰动 adapter 和响应预测头。
- **微调组**：使用相同输入和预测头，额外以较小学习率更新编码器最后两层 Transformer。

两组使用相同的五折划分、验证规则和指标。官方 checkpoint 的编码器 153/153 个参数张量均
成功载入；只转换了注意力层的 Q/K/V 权重存储形式。105 个扰动基因中，102 个可以映射到
预训练词表；3 个词表外基因单独记录并使用可训练 fallback embedding。这是**预训练编码器
迁移实验**，并非完整复刻 scGPT 论文中所有扰动预测设置。实现和来源见
[`src/mini_vc/models/scgpt_transfer.py`](src/mini_vc/models/scgpt_transfer.py) 与
[`src/mini_vc/scgpt_core/README.md`](src/mini_vc/scgpt_core/README.md)。

## 6. 项目结构

```text
k562-perturbation-benchmark/
├── README.md
├── environment.yml
├── requirements.txt
├── configs/                 # V0.1、五折与 scGPT 实验配置
├── data/
│   ├── raw/                 # 下载数据；Git 忽略大文件
│   ├── processed/           # 可重建的预处理数据；Git 忽略
│   └── README.md            # 数据 provenance
├── reports/                 # 数据、方法和结果报告
├── results/                 # CSV、指标和图；大型训练产物仅在本地
├── scripts/                 # 命令行入口
├── src/mini_vc/
│   ├── data/                # QC、split、pseudobulk、dataset
│   ├── models/              # baselines、MLP、Mini-VC、scGPT 迁移头
│   ├── scgpt_core/          # 官方 scGPT 0.2.4 的最小 MIT 源码子集
│   ├── training/            # 训练循环、loss、checkpoint
│   └── evaluation/          # 指标、artifact、结果图
└── tests/                   # split、模型、评价和 scGPT 输入测试
```

## 7. 环境安装

本项目在 Windows、Python 3.11、RTX 4060 和 CUDA 版 PyTorch 上完成实际运行。

### Anaconda / Miniconda

首次使用，在仓库根目录创建实验环境：

```bash
conda env create -f environment.yml
conda activate mini_vc
```

已有 `mini_vc` 环境时直接激活，必要时安装依赖：

```bash
conda activate mini_vc
python -m pip install -r requirements.txt
```

检查 PyTorch 和 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

配置中的 `device: "auto"` 会优先使用 CUDA；没有可用 GPU 时仍能使用 CPU，但完整训练会更慢。
`requirements.txt` 固定了本次复现实验的依赖版本和 CUDA 13.0 PyTorch wheel 来源。
scGPT 编码器使用仓库中的最小源码子集，checkpoint 由 `gdown` 下载。
原始数据、处理矩阵、预训练权重和预测数组不随仓库分发；最终 CSV、指标和图保留在仓库中。

其他硬件可先安装与系统匹配的 PyTorch，再安装 `requirements-common.txt`。例如 CPU 环境：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-common.txt
```

CPU 安装路径没有用于本次完整实验。单元测试不需要下载数据或预训练 checkpoint：

```bash
python -m unittest discover -s tests -v
```

## 8. 从零复现

以下命令均从仓库根目录执行。数据、模型参数和预测数组在本地生成；
最终数值结果和图可以直接在仓库中查看。

### 8.1 下载并核验数据

```bash
python scripts/download_data.py
```

已存在且校验正确时不会重复下载；中断后会保留 `.part` 文件并尽量续传。

### 8.2 检查原始数据

```bash
python scripts/inspect_data.py
```

输出：`reports/data_summary.md`。

### 8.3 QC、normalization、HVG、split 与 pseudobulk

```bash
python scripts/preprocess.py --config configs/preprocess.json
```

输出：`data/processed/norman_v0_1/` 和 `reports/preprocessing_summary.md`。

### 8.4 训练基线

```bash
python scripts/train.py --config configs/baseline.json
python scripts/summarize_baselines.py
```

### 8.5 训练 Mini-VC

```bash
python scripts/train.py --config configs/mini_vc.json
python scripts/summarize_mini_vc.py
```

训练固定随机种子，自动选择 GPU/CPU，使用 validation response MSE 进行学习率调度和 early
stopping，并保存 best checkpoint、逐 epoch 日志、测试预测和逐条件指标。测试集不参与模型选择。

### 8.6 独立复算一个预测文件的指标

```bash
python scripts/evaluate.py --predictions results/mini_vc/unseen_combo/mini_vc/predictions_test.npz --metrics results/mini_vc/unseen_combo/mini_vc/metrics.json
```

该命令从保存的预测重新计算指标，并检查是否与训练时写出的 `metrics.json` 一致。

### 8.7 生成结果图和分析报告

```bash
python scripts/analyze_results.py
```

该脚本只读取冻结的测试预测，不重新训练或调参。输出位于 `results/figures/`，报告为
[`reports/stage9_results_analysis.md`](reports/stage9_results_analysis.md)。

### 8.8 复现 V0.2 五折与 scGPT 比较

以下命令按顺序执行。第一次下载 checkpoint 约 197 MiB；
脚本会校验 SHA256，已有正确文件会直接复用。

```bash
python scripts/download_scgpt.py
python scripts/preprocess.py --config configs/preprocess_scgpt_folds.json
python scripts/audit_scgpt_folds.py
python scripts/train.py --config configs/baseline_scgpt_folds.json
python scripts/train.py --config configs/mini_vc_scgpt_folds.json
python scripts/train_scgpt.py --config configs/scgpt_folds.json
python scripts/summarize_scgpt_folds.py
python scripts/verify_scgpt_checkpoint.py
python scripts/analyze_model_performance.py
```

可选的原始 `seen` 和 `unseen_gene` 划分补充实验：

```bash
python scripts/train_scgpt.py --config configs/scgpt_secondary.json
python scripts/summarize_scgpt_secondary.py
```

主要输出为 [`reports/scgpt_fivefold_results.md`](reports/scgpt_fivefold_results.md)、
[`reports/scgpt_secondary_results.md`](reports/scgpt_secondary_results.md) 和
[`五折汇总 CSV`](results/scgpt_folds/model_comparison_131_conditions.csv)。后者由逐条件测试指标汇总，
[`逐条件指标`](results/scgpt_folds/all_models_per_condition.csv) 和
[`配对差异`](results/scgpt_folds/paired_differences.csv) 也随仓库保存。大型逐条件预测和模型参数不会
进入 Git，需要在本地按上述命令生成。`verify_scgpt_checkpoint.py` 会重载官方权重与训练好的
adapter，并核对其测试预测；可用 `--scheme` 和 `--mode` 检查其他划分与模式。

### 8.9 加入 Reactome 通路知识（V0.3）

沿用五折数据和训练配置：

```bash
python scripts/download_reactome.py
python scripts/train_reactome.py
python scripts/summarize_reactome.py
```

模型保留原来的 105 个扰动基因身份特征，另外加入 130 组不同通路成员模式的“至少一个基因参与”和
“两个基因共同参与”特征。外部知识来自
[Reactome 人类通路基因集](https://reactome.org/download/current/ReactomePathways.gmt.zip)，
不是本实验测得的蛋白数量。相同五折上训练 Ridge、MLP，并各做 3 次随机打乱通路对应关系的
对照。详细结果见 [`reports/reactome_pathway_results.md`](reports/reactome_pathway_results.md)。

## 9. 评价指标

同时评价绝对表达和相对 control 的响应：

- all-gene MSE / RMSE、Pearson、Spearman；
- expression delta 的 Pearson、Spearman 和 cosine similarity；
- true differential-response genes 上的 MSE 与相关性；
- top-20、top-50、top-100 响应基因 recovery；
- 每个 perturbation 的指标，以及对条件做 macro average 的总体指标。

这里的 DE 排名依据响应幅度，不是差异表达显著性检验。
No-change 的 delta 恒为零，response correlation 和 DE ranking 报告为空值。

## 10. V0.1 结果

以下均为冻结测试集结果，不是训练指标：

| Split | 模型 | All-gene MSE ↓ | Response Pearson ↑ | Top-100 recovery ↑ |
|---|---|---:|---:|---:|
| seen | Ridge | 0.0036 | 0.7641 | 0.5654 |
| seen | MLP | **0.0033** | **0.7681** | **0.5731** |
| seen | Mini-VC | 0.0035 | 0.7454 | 0.5596 |
| unseen_combo | Ridge | 0.0017 | 0.9006 | 0.7485 |
| unseen_combo | MLP | **0.0017** | **0.9188** | **0.7523** |
| unseen_combo | Mini-VC | 0.0019 | 0.9142 | 0.7431 |
| unseen_gene | Ridge | **0.0095** | **0.5224** | 0.4926 |
| unseen_gene | MLP | 0.0109 | 0.4584 | **0.5007** |
| unseen_gene | Mini-VC | 0.0113 | 0.4221 | 0.4559 |

主要结论是：**Mini-VC 学到了可用的 latent transition，但没有稳定超过 Ridge/MLP。**
在主要的 unseen-combination 任务中，它与 MLP 接近；Mini-VC 的 top-20 recovery 为 0.7769，
高于 MLP 的 0.7423，但总体 MSE 和宏平均 response Pearson 仍由 MLP 最好。逐条件分析中，
Mini-VC 在 13 个组合中的 6 个取得最高 Pearson。

Ridge 的表现与加性近似适合这个 pseudobulk 任务的解释相容，但模型排名不能单独确定
真实生物互作机制。Mini-VC 在这三个划分上没有获得一致的性能优势。完整数字和图见
[`reports/stage8_training_summary.md`](reports/stage8_training_summary.md) 与
[`reports/stage9_results_analysis.md`](reports/stage9_results_analysis.md)。

### V0.2：131 个双基因组合五折测试

以下各项均为相同 131 个未见双基因组合上的**逐条件宏平均**；↑/↓ 表示越高/越低越好。

| 模型 | Response Pearson ↑ | All-gene MSE ↓ | Top-100 recovery ↑ |
|---|---:|---:|---:|
| Ridge | **0.8963** | 0.002015 | 0.7204 |
| MLP | 0.8913 | 0.001982 | **0.7224** |
| Mini-VC | 0.8837 | **0.001935** | 0.7163 |
| scGPT 编码器冻结 | 0.8816 | 0.002111 | 0.7102 |
| scGPT 编码器部分微调 | 0.8817 | 0.002068 | 0.7129 |

微调相对冻结组的 response Pearson 仅增加 0.0001，131 条件配对 bootstrap 区间为
`[-0.0031, 0.0034]`。微调组相对 MLP 低 0.0096，区间为 `[-0.0165, -0.0040]`。
因此目前的证据不支持“加入 scGPT 便会提升此任务表现”；13 条件结果中的模型排名也不能
直接推广到全部组合。不同指标对应的最优模型不同，完整逐条件结果、图和局限见
[`reports/scgpt_fivefold_results.md`](reports/scgpt_fivefold_results.md)。

原始 `seen` 划分中，冻结/微调 scGPT 的 Pearson 分别为 0.7630/0.7634；原始
`unseen_gene` 划分中分别为 0.4450/0.4641，后者仍低于 Ridge 的 0.5224。
这些是已观察过的原始划分，属于补充分析。

### V0.3：Reactome 通路知识对照

在同样 131 个五折测试组合上，加入通路特征后，Ridge 的响应 Pearson 从 **0.8963** 降至
**0.8784**，MLP 从 **0.8913** 降至 **0.8640**。通路覆盖 105 个扰动基因中的 66 个；
正确的基因—通路对应关系也没有稳定超过随机打乱的对应关系。因此，这种静态通路特征
**没有改善当前任务**。这不能说明蛋白信息没有用，因为通路成员身份不等于 K562 中蛋白的
含量、修饰或活性。完整逐组合比较见
[`reports/reactome_pathway_results.md`](reports/reactome_pathway_results.md)。

## 11. 测试

项目使用 Python 标准库 `unittest`，无需额外安装测试框架：

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
```

当前共有 25 项测试，覆盖 split 隔离、五折覆盖、双扰动顺序不变性、未知扰动 mask、
Reactome 通路特征、模型前向/反向、scGPT 基因映射与分箱、指标、DE recovery、
诊断误差分解和预测数据对齐。

## 12. 推荐阅读顺序

主要实现入口：

1. `reports/prediction_task.md`：先理解统计任务和为什么没有单细胞前后配对。
2. [dataset.py](src/mini_vc/data/dataset.py)：reference、delta 和 multi-hot。
3. [splits.py](src/mini_vc/data/splits.py)：原始划分和五折划分。
4. [mini_vc.py](src/mini_vc/models/mini_vc.py)：Encoder、扰动 embedding、Transition、Decoder。
5. [losses.py](src/mini_vc/training/losses.py)：三个训练目标。
6. [metrics.py](src/mini_vc/evaluation/metrics.py)：条件质心和宏平均评价。

## 13. 当前局限

- 只建模 2,000 个 HVG 的平均响应，不生成单细胞分布。
- 数据只有 K562 和 CRISPRa，不能直接外推到其他细胞类型或扰动方式。
- reference 和 target 是独立细胞的 pseudobulk，不是纵向时间轨迹。
- V0.1/V0.2 的主模型只使用扰动身份；V0.3 测试了静态 Reactome pathway 特征，
  但仍没有 K562 的蛋白测量、GRN 或 DNA sequence 信息。
- 双扰动 embedding 求和隐含简单可组合性，不能完整表示高阶 genetic interaction。
- identity-only MLP 和 Mini-VC 屏蔽未知基因；scGPT 提供多数基因的预训练 token，但未见基因的补充测试不足以确认广泛迁移能力。
- 没有外部数据集验证或预测不确定性估计；seen 留出的批次仍属于同一次实验。
- 基础测试结果已在五折和 Reactome 比较前观察过，后两组属于同一数据集上的扩展分析。
- scGPT 迁移实验每个批次只编码 4 个 control cells、512 个输入基因，新建的扰动预测头需从头
  学习；此负结果不等于排除所有可能的 scGPT 架构或输入预算。
- 五折配对区间仅描述本数据集中条件间的变化；条件之间共享实验批次和组成基因，不能当作
  独立外部人群的显著性检验。

## 14. 仓库内容

仓库包含源码、固定配置、测试、方法与结果报告，以及最终 CSV、指标 JSON 和图。
大型数据和模型产物按上面的命令下载或重建。旧报告文件名中的阶段编号保留，以便对应原始实验。
准备上传的文件范围见 [UPLOAD.md](UPLOAD.md)。

## 15. 引用与许可

使用该数据或结果时，至少引用 Norman et al. (2019)、GEO `GSE133344` 和 scPerturb；
使用 V0.2 的预训练模型还应引用 [scGPT 论文](https://www.nature.com/articles/s41592-024-02201-0)
及[官方项目](https://github.com/bowang-lab/scGPT)。
使用 V0.3 的通路知识还应注明 [Reactome 数据来源](https://reactome.org/download-data)。
数据来源和许可说明见 [data/README.md](data/README.md)。
`src/mini_vc/scgpt_core/` 保留上游 MIT 许可证和来源说明；项目自有代码尚未指定开源许可证。

代码与文档的编写、整理使用了 Codex。实验设置和数值结果见配置、日志、逐条件指标和报告。
