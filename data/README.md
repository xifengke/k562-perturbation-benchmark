# Data

本目录只保存可重新获取或重新生成的数据。大型表达矩阵不应提交到 Git。

## 目录

- `raw/`：从可信公开来源下载、未经本项目修改的数据。
- `processed/`：由本项目预处理脚本生成的数据。
- `external/reactome/`：V0.3 使用的公开人类通路知识及来源记录。

## V0.3 Reactome 通路知识

通过 `python scripts/download_reactome.py` 下载官方
[`ReactomePathways.gmt.zip`](https://reactome.org/download/current/ReactomePathways.gmt.zip)。
本次文件为 `298,479` bytes；SHA256 为
`8c1dbc8578431da5d2d5118262718c60b553a9be3398e93658daa069e4a9afd4`。
下载时间和校验值记录在 `external/reactome/download_manifest.json`。
Reactome 的 `current` 地址会更新，重现实验时应核对 SHA256；如果文件更新，
新版本需要单独记录和评估。

这份 GMT 是基因与人工整理通路之间的静态关联，**不是 K562 的蛋白表达或活性测量**。
原始 Norman AnnData 没有与这些细胞配对的蛋白组数据。
Reactome 的[许可页面](https://reactome.org/license)说明数据库数据采用 CC0；
发布分析时仍建议引用 Reactome，并保留下载文件与版本信息。

## V0.1 数据源

本项目选择 Norman 等人 2019 年的 K562 CRISPRa Perturb-seq 数据，并使用
scPerturb 提供的统一 AnnData 版本。

| 字段 | 值 |
|---|---|
| 文件 | `NormanWeissman2019_filtered.h5ad` |
| 数据集 | Norman et al. 2019 / NormanWeissman2019 |
| 原始论文 | *Exploring genetic interaction manifolds constructed from rich single-cell phenotypes* |
| 论文 DOI | `10.1126/science.aax4438` |
| GEO accession | `GSE133344` |
| 原始 GEO | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344 |
| 整理来源 | scPerturb RNA collection |
| scPerturb 版本 | `1.4` |
| Zenodo record | `10.5281/zenodo.13350497` |
| 下载地址 | https://zenodo.org/records/13350497/files/NormanWeissman2019_filtered.h5ad?download=1 |
| 本次下载时间 | `2026-09-23 09:19:05 UTC`（`2026-09-23 17:19:05 Asia/Shanghai`） |
| 预期大小 | `698,680,199` bytes（约 698.7 MB / 666.3 MiB） |
| 预期 MD5 | `c870e6967d91c017d9da827bab183cd6` |
| 数据形状（公开记录） | `111,445 cells × 33,694 genes` |
| 数据类型 | K562 single-cell RNA counts；control、单基因和双基因 CRISPRa |

实际下载时间、文件大小和校验值由下载脚本写入
`raw/NormanWeissman2019_filtered.download.json`，不要把未完成的 `.part` 文件当成数据集。

## 下载

从仓库根目录运行：

```bash
python scripts/download_data.py
```

下载器具有以下行为：

- 已存在且校验通过时跳过下载；
- 使用 `.part` 临时文件并尽可能通过 HTTP Range 续传；
- 完成后检查精确字节数和 MD5；
- 只有校验通过后才将临时文件改为正式文件名；
- 写出一个小型 JSON provenance manifest。

若已存在损坏文件，脚本会停止并提示错误，不会静默覆盖。确认要重新下载时可运行：

```bash
python scripts/download_data.py --force
```

## 为什么不直接使用其他版本

- GEO 官方 filtered MTX 加 metadata 约需下载 1.1 GB，并需要手动组装 AnnData。
- GEO 的 `GSE133344_RAW.tar` 约 10 GB，不适合以模型学习为重点的 V0.1。
- CPA 教程版已经筛选到 5,044 个基因并完成 normalization/HVG，方便训练但隐藏了本项目希望学习和审计的预处理步骤。

## 许可与引用

GEO/NCBI 公开提供该数据。NCBI 对其分子数据库数据不主动施加使用和分发限制，
但不能代替原始提交者授予潜在权利。Zenodo 页面未明确显示这份上游数据的独立许可证；
scPerturb 代码的 MIT 许可证不能自动解释为数据许可证。

使用结果时至少应引用：

1. Norman TM et al. *Science* (2019), DOI `10.1126/science.aax4438`。
2. GEO accession `GSE133344`。
3. Peidli S, Green TD, et al. scPerturb, *Nature Methods* (2024)，以及 Zenodo record `13350497`。
