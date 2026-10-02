# Preprocessing 与 split 报告

## QC

- 输入细胞：111,445
- 保留细胞：91,130
- 去除细胞（条件并集）：20,315
- `good_coverage=False`：6,938
- `number_of_cells != 1`：20,277
- counts < 1000：0
- genes < 500：0
- mitochondrial percentage > 20.0：54

这些失败原因可能重叠，所以各项不能直接相加。`number_of_cells > 1` 被当作 multiplet，
`number_of_cells == 0` 与低 guide coverage 失败相符。

## Normalization 与 HVG

- 每个细胞先按 library size 缩放至 10000 counts；
- 对每个非零值执行 `log1p`；
- 每个 split scheme 只用其 train cells 拟合 gene mean/variance/HVG；
- 选择 2000 genes，并强制保留实验涉及且存在于矩阵中的 perturbation genes；
- 105 个 perturbation genes 中有 102 个能按 gene symbol 在矩阵中匹配；未匹配：`C19orf26, C3orf72, KIAA1804`；
- 5 套 train-only HVG 集合的共同基因数：1711。

## Split 与 pseudobulk 数量

单元格顺序均为 train / validation / test。

| scheme | cells | pseudobulk samples | genes | forced perturbation genes |
|---|---:|---:|---:|---:|
| unseen_combo_fold0 | 76342 / 4847 / 9941 | 1576 / 112 / 224 | 2000 | 102 |
| unseen_combo_fold1 | 79734 / 3148 / 8248 | 1584 / 112 / 216 | 2000 | 102 |
| unseen_combo_fold2 | 76599 / 4861 / 9670 | 1584 / 112 / 216 | 2000 | 102 |
| unseen_combo_fold3 | 78974 / 4115 / 8041 | 1584 / 112 / 216 | 2000 | 102 |
| unseen_combo_fold4 | 79356 / 3831 / 7943 | 1584 / 112 / 216 | 2000 | 102 |

每个 pseudobulk 是同一 `(split, perturbation, gemgroup)` 中所有 QC-passing cells 的
log-normalized expression 均值。不存在 control–perturbed 单细胞伪配对。

## 输出文件

- `cell_splits.csv.gz`：每个原始 cell 的 QC 与各划分的 split assignment；
- `<scheme>/expression.npy`：`n_pseudobulk × 2000` float32 expression；
- `<scheme>/samples.csv`：行 metadata；
- `<scheme>/genes.csv`：列 metadata 和 train-only HVG statistics；
- `<scheme>/split_details.json`：held-out batches/conditions/genes；
- `manifest.json`：配置和尺寸汇总。
