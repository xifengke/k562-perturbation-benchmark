# Norman 2019 数据探索报告

本报告由 `scripts/inspect_data.py` 从真实 H5AD 文件自动生成。检查过程使用只读
backed 模式，没有 normalization、HVG selection 或 train/test split。

## 一眼结论

- 矩阵：**111,445 cells × 33,694 genes**。
- 扰动标签：**237** 个，其中 control 11,855 个细胞。
- 条件组成：1 个 control、105 个单基因条件、131 个双基因条件。
- 数据包含 **8 个 `gemgroup` batches**。
- 矩阵稀疏度约 **90.37%**。
- 数值尺度判断：**raw counts**。
- 类别明显不均衡：最大条件 / 最小条件为 **219.5×**；去掉 control 后仍为 **36.3×**。

## 基本结构

| 项目 | 值 |
|---|---:|
| AnnData shape | `111,445 × 33,694` |
| HDF5 matrix shape | `111,445 × 33,694` |
| 稀疏格式 | `csc_matrix` |
| 非零元素 nnz | 361,582,621 |
| 总矩阵元素 | 3,755,027,830 |
| 稀疏度 | 90.3707% |
| 存储 dtype | `float32` |
| cell index 重复数 | 0 |
| gene index 重复数 | 0 |
| guide-level labels | 290 |
| harmonized perturbation labels | 237 |
| `.raw` | `False` |
| layers | `[]` |
| `uns` keys | `[]` |

## 一行、一个样本和一个基因分别是什么？

- **一行**是一个经过 GEO/scPerturb 过滤、并成功获得转录组测量的 K562 细胞。
- **一个观测样本**在 cell-level 数据中就是一个细胞；但它不是某个 control cell 的“扰动后版本”。
- **一列**是一个基因。矩阵元素是该细胞中检测到的该基因 UMI count。
- 建模时会从互不重叠的细胞集合构建 condition-level/pseudobulk 样本，减少 dropout，且不制造假配对。

## Perturbation label 与 control

`perturbation` 是 scPerturb 整理后的基因级扰动标签。单基因标签类似 `KLF1`，
双基因标签类似 `CEBPE_RUNX1T1`。`guide_id` 保留更细的 guide/cassette 身份，
因此本文件有 290 个 guide-level labels，但只有
237 个 gene-level perturbation labels。

control 标签为 `control`，表示两个位置均使用
non-targeting guide 的细胞。单基因条件由目标 guide 加 negative-control guide 构成，
双基因条件的两个位置均为目标基因 guide。

| 类型 | 条件数 | 细胞数 | 细胞占比 |
| --- | --- | --- | --- |
| control | 1 | 11,855 | 10.64% |
| single | 105 | 57,831 | 51.89% |
| double | 131 | 41,759 | 37.47% |

同一 `perturbation` 同时映射到多个 `nperts` 值的异常标签数：**0**。

## Metadata columns

### Cell metadata (`adata.obs`)

| 字段 | dtype |
| --- | --- |
| guide_id | category |
| read_count | int64 |
| UMI_count | int64 |
| coverage | float64 |
| gemgroup | int64 |
| good_coverage | bool |
| number_of_cells | int64 |
| tissue_type | category |
| cell_line | category |
| cancer | bool |
| disease | category |
| perturbation_type | category |
| celltype | category |
| organism | category |
| perturbation | category |
| nperts | int64 |
| ngenes | int64 |
| ncounts | float64 |
| percent_mito | float64 |
| percent_ribo | float64 |

### Gene metadata (`adata.var`)

| 字段 | dtype |
| --- | --- |
| ensemble_id | str |
| ncounts | float64 |
| ncells | int64 |

## Counts、normalization 与 log1p 检查

高度符合未归一化 UMI counts：非零值为非负整数，细胞总 counts 明显不恒定，且对象没有 `log1p` 标记或额外 layer。虽然矩阵以 float32 存储，数值仍是整数 counts。

检查证据：

- 从 `X/data` 均匀抽取 900,000 个非零值；
- sampled nonzero min/max：1.0 / 95.0；
- integer-like fraction：100.000000%；
- negative fraction：0.000000%；
- `uns['log1p']` 是否存在：`False`；
- `layers`：`[]`；
- cell total-count coefficient of variation：0.410。

注意：“raw counts”在这里指 Cell Ranger 之后的 UMI count matrix，不是 FASTQ，也不是完全未经
cell filtering 的测序数据。后续仍需做 QC、library-size normalization 和 `log1p`。

### 每个细胞的 total counts

| 分位点 | ncounts |
| --- | --- |
| 最小值 | 3,679.0 |
| 1% | 4,383.9 |
| 25% | 10,706.0 |
| 中位数 | 13,855.0 |
| 75% | 17,873.0 |
| 99% | 32,714.2 |
| 最大值 | 64,352.0 |

### 每个细胞检测到的 genes

| 分位点 | ngenes |
| --- | --- |
| 最小值 | 974.0 |
| 1% | 1,589.0 |
| 25% | 2,774.0 |
| 中位数 | 3,233.0 |
| 75% | 3,717.0 |
| 99% | 5,079.0 |
| 最大值 | 6,760.0 |

## Batch 信息

`gemgroup` 对应合并数据中的 8 个 10x Chromium lane/gem groups。各组规模相近，但不能因此
假定没有 batch effect；后续 split 和可视化都应保留该字段。

| gemgroup | 细胞数 | 占比 |
| --- | --- | --- |
| 1 | 15,033 | 13.49% |
| 2 | 13,787 | 12.37% |
| 3 | 14,246 | 12.78% |
| 4 | 13,083 | 11.74% |
| 5 | 13,101 | 11.76% |
| 6 | 13,792 | 12.38% |
| 7 | 14,137 | 12.69% |
| 8 | 14,266 | 12.80% |

## Guide coverage

`good_coverage=False` 的细胞仍存在于这份 filtered H5AD 中，共
6,938 个。该字段与 `number_of_cells` 的含义和是否用于
额外 QC，应结合原始处理说明决定，不能看到 “filtered” 就默认全部通过 guide QC。

| good_coverage | 细胞数 | 占比 |
| --- | --- | --- |
| True | 104,507 | 93.77% |
| False | 6,938 | 6.23% |

## 类别不均衡

- 每个 perturbation 的细胞数：最少 54，中位数 354.0，最多 11,855。
- 最大类是 control，11,855 cells，占 10.64%。
- 最大/最小条件比例为 219.5×；排除 control 后为 36.3×。
- 因此训练 loss 不能让大条件无限主导，评价也必须先按 perturbation 分别计算再做 macro average。

细胞数最多的 15 个条件：

| perturbation | 细胞数 | 占全部细胞 |
| --- | --- | --- |
| control | 11,855 | 10.64% |
| KLF1 | 1,960 | 1.76% |
| BAK1 | 1,457 | 1.31% |
| CEBPE | 1,233 | 1.11% |
| CEBPE_RUNX1T1 | 1,219 | 1.09% |
| UBASH3B | 1,202 | 1.08% |
| ETS2 | 1,201 | 1.08% |
| TBX3_TBX2 | 1,167 | 1.05% |
| OSR2 | 1,003 | 0.90% |
| SLC4A1 | 1,000 | 0.90% |
| SET | 986 | 0.88% |
| ELMSAN1 | 937 | 0.84% |
| ETS2_CNN1 | 905 | 0.81% |
| MAP2K6 | 878 | 0.79% |
| FOXF1 | 874 | 0.78% |

## 预测任务

给定 reference 和扰动身份，预测条件平均响应：

> 给定独立估计的 K562 control/reference state 与一个单基因或双基因 CRISPRa identity，
> 预测该 perturbation 条件下的平均表达和相对 control 的 `delta expression`。

训练和测试细胞互不重叠。unseen-combination 留出完整双基因条件，
同时将对应单基因条件保留在训练集，用于评价组合泛化。

## 预处理选择

1. 稀疏读取表达矩阵，按块处理，避免将整个矩阵转为 dense。
2. 使用原始 counts 逐细胞进行 library-size normalization 和 `log1p`。
3. 保留 `gemgroup`，在同批次内构建 reference 和 target pseudobulk。
4. 每个划分用训练细胞选择 2,000 个 HVG，并保留可匹配的扰动目标。
5. 记录 cell ID 和 condition 的 split assignment；HVG 由训练细胞拟合，测试 DE 排名由测试 delta 构成。
