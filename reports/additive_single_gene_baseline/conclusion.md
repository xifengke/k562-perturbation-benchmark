# 五折单基因响应相加基线

## 评估口径

每条测试记录的预测表达为 `test_control + (train_A - train_control) + (train_B - train_control)`。两个单基因和训练对照都取自当前折训练集的同一 GEM group，最后加回该记录所在 GEM group 的测试对照。使用每折原有的 2,000 个基因和原评估函数，按组合等权汇总各 GEM group 的表达，再评分。总分对 131 个组合做宏平均。Top-100 recovery 按响应绝对值排序。

逐折核验了测试样本 ID、扰动标签、真实表达和对照矩阵。训练 checkpoint 的基因名与 `genes.csv` 顺序一致。从 Mini-VC 保存预测复算的逐条件和逐折指标与历史文件一致，逐条件浮点相关性容差为 3×10⁻⁷。

## 131 个组合的配对比较

| 指标 | 单基因相加 | Mini-VC | 相加胜 | Mini-VC 胜 | 持平 |
|---|---:|---:|---:|---:|---:|
| 全基因 MSE ↓ | 0.002855 | 0.001935 | 37 | 94 | 0 |
| 响应 Pearson ↑ | 0.887543 | 0.883719 | 58 | 73 | 0 |
| Top-100 recovery ↑ | 0.706565 | 0.716260 | 57 | 68 | 6 |

Mini-VC 的宏平均全基因 MSE 下降 32.2%，Top-100 recovery 增加 0.0097。相加基线的宏平均响应 Pearson 高 0.0038。

## 组合差异

- 单基因相加在 27 个组合上三项均胜。
- Mini-VC在 58 个组合上三项均胜。
- 按下列描述性阈值，三项同时接近的组合有 0 个。
- 全基因 MSE 相对差距 ≤5%的组合有 9 个，例如 FOXL2_HOXB9、JUN_CEBPB、KLF1_MAP2K6。
- 响应 Pearson 差距 ≤0.01的组合有 25 个，例如 CDKN1C_CDKN1A、CDKN1C_CDKN1B、CEBPB_OSR2。
- Top-100 recovery 差距 ≤0.02的组合有 34 个，例如 AHR_FEV、BPGM_SAMD1、CBL_TGFBR2。

下表列出每项指标上各模型优势最大的三个组合。MSE 使用绝对下降量，其他指标使用绝对增量。逐组合 CSV 中 `mini_vc_advantage_*` 的正值支持 Mini-VC，负值支持相加基线。

| 指标 | 单基因相加优势最大 | Mini-VC 改善最大 |
|---|---|---|
| 全基因 MSE ↓ | ZBTB10_ELMSAN1（0.003150）、KLF1_COL2A1（0.002842）、LHX1_ELMSAN1（0.002503） | CEBPE_CEBPA（0.017744）、CEBPB_CEBPA（0.014924）、SET_CEBPE（0.007729） |
| 响应 Pearson ↑ | LHX1_ELMSAN1（0.275190）、TMSB4X_BAK1（0.272240）、KLF1_COL2A1（0.240627） | CBL_UBASH3A（0.120590）、PLK4_STIL（0.116813）、MAPK1_PRTG（0.108450） |
| Top-100 recovery ↑ | PTPN12_SNAI1（0.180000）、AHR_KLF1（0.170000）、MAP2K3_ELMSAN1（0.150000） | PTPN12_UBASH3A（0.190000）、ETS2_IGDCC3（0.170000）、MAPK1_PRTG（0.170000） |

## 复算与边界

在安装仓库依赖的 Python 环境中，从仓库根目录运行以下命令。

```powershell
python -B reports/additive_single_gene_baseline/evaluate_additive.py
```

`per_condition.csv` 保存完整配对分数，`close_*` 列标记接近的组合。`sample_matches.csv` 记录每条测试记录及其两个训练单基因和对照的样本 ID。样本 ID 在折内解释。`per_fold.csv` 和 `overall.csv` 保存全部原评估指标。

结果适用于本次 Norman K562 CRISPRa 五折测试中，同批次两个单基因响应均可用的未见组合。这里的相加是在 log-normalized pseudobulk 表达空间中进行，性能差异只提供该任务中的预测比较。组合的相加拟合优劣本身不足以推断生物互作。
