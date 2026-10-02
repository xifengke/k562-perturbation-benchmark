# GitHub 文件清单

项目已完成。公开仓库包含代码、实验设置和结果证据，大型数据和模型文件通过复现命令获取。

## 上传

- 根目录：`README.md`、`UPLOAD.md`、`.gitignore`、`environment.yml`、`requirements.txt`、`requirements-common.txt`。
- `src/`、`scripts/`、`configs/`、`tests/` 的源码和配置，不包含缓存。
- `data/README.md`。
- `reports/` 中的数据、方法、结果报告，以及 `scgpt_fold_audit.json`。
- `results/` 中的最终 CSV：汇总表、逐条件指标、配对差异、通路覆盖统计和训练日志。
- 各模型的 `metrics.json`。
- `results/figures/` 中的 PNG 和 `analysis_summary.json`。
- `reports/model_performance_analysis.md` 和 `results/model_diagnostics/` 中的诊断 CSV。

`.gitignore` 已按上述范围设置。逐条件 CSV 是小型数值证据；完整预测数组和 checkpoint 不包含在上传范围内。

## 留在本地

- `.venv/`、Python 缓存和编辑器文件。
- `data/raw/`、`data/processed/`、`data/external/` 中的文件，包括带本机路径的下载清单。
- `checkpoints/`、`vendor/`、`logs/`。
- `results/` 中的 `.pt`、`.npz` 和 input/pathway audit JSON；名字包含 `partial` 的 CSV。
- `reports/stage10_completion.md` 和 `reports/scgpt_project_completion.md`：历史开发记录。
- `notebooks/`：只有占位说明，没有实际 notebook。
- 本地凭据和 `.env` 文件。

排除的文件保留在本地，没有删除。分析和预测检查脚本需要的数据由 README 中的复现流程生成。

## 发布前核对

1. 初始化仓库后，先查看待提交文件清单，确认没有大数据、模型权重或本地凭据。
2. 运行 `python -m unittest discover -s tests -v`，以及 `python -m compileall -q src scripts tests`。
3. 确认 README 和报告中的图、CSV 与源码链接均包含在上传范围内。
4. 自有代码尚未指定开源许可证。若决定授予开源使用权限，需要添加根目录 `LICENSE`，并同步 README；第三方 scGPT 的 MIT 许可证继续保留。

代码和文档的编写、整理使用了 Codex。实验设置和数值结果见配置、日志、逐条件指标和报告。
