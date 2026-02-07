# Funding Cooperation 分析脚本（全量重跑版）

该版本不再使用断点续跑。

## 运行

```bash
python analysis_pipeline.py \
  --file1 /data/student2601/WOS/WOS/Funding/merged_paper_info_with_details_full.tsv \
  --file2 /data/student2601/WOS/WOS/Funding/2022merged_paper_info_new_unique.csv \
  --out_dir ./output_funding_analysis \
  --n_jobs 0
```

- `--n_jobs 0` 表示自动使用 `CPU核数-1` 并行处理地址/作者解析，以加速大文件运行。

## 本次关键修正

- **全量重跑**：不再依赖中间断点文件。
- **双文件合并**：自动识别分隔符，兼容 `.tsv` + `.csv`，确保 2022/2023 数据并入。
- **美国归并**：`USA / United States / U.S.A` 等归并为 `usa`，避免 Top 国家漏计。
- **Gephi 导出**：输出国家网络与机构网络点边表至 `output_funding_analysis/gephi/`。

## 输出目录

- `output_funding_analysis/metrics/`：指标 CSV
- `output_funding_analysis/figures/`：图表 PNG
- `output_funding_analysis/gephi/`：Gephi 点/边 CSV


## 单独可视化重绘（图6/图7优化）

如果你已经有 `metrics` 结果，可直接运行：

```bash
python metrics_visualization_refined.py   --metrics_dir /data/student2601/WOS/WOS/0207/output_funding_analysis/metrics   --out_dir /data/student2601/WOS/WOS/0207/output_funding_analysis/figures_refined
```

该脚本会：
- 基于 `metric_partner_year.csv` 重算 Top20，避免沿用旧 top20 文件里的异常项。
- 对合作国家做别名归并（含 USA）并过滤明显噪声项（如 `wei/jing/yu` 等误识别词）。
- 输出优化版图：`fig6_top20_partners_refined.png`、`fig7_partner_heatmap_refined.png`。


## 全量美化重绘（基于现有 metrics）

可直接对 `output_funding_analysis/metrics` 重绘全部图（不重跑原始数据处理）：

```bash
python all_visualizations_refined.py   --metrics_dir /data/student2601/WOS/WOS/0207/output_funding_analysis/metrics   --out_dir /data/student2601/WOS/WOS/0207/output_funding_analysis/figures_beautified
```

输出：
- 11 张美化图（`01_...png` 到 `11_...png`）
- `cleaned_tables/partner_year_cleaned.csv` 与 `cleaned_tables/top20_cleaned.csv`（用于核查 USA 是否聚合、异常国家是否被过滤）
