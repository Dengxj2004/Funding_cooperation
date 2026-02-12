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


## 重新统计美国合作次数并重绘图6/图7（修复邮编+USA地址）

你发现的地址形式（如 `NJ 08544 USA`）非常关键。可运行下面脚本重新统计：

```bash
python recount_us_and_redraw_fig6_fig7.py   --file1 /data/student2601/WOS/WOS/Funding/merged_paper_info_with_details_full.tsv   --file2 /data/student2601/WOS/WOS/Funding/2022merged_paper_info_new_unique.csv   --out_dir /data/student2601/WOS/WOS/0207/output_funding_analysis/figures_recount   --metrics_out_dir /data/student2601/WOS/WOS/0207/output_funding_analysis/metrics_recount   --n_jobs 0
```

输出：
- 图：`fig6_top20_partners_recount.png`、`fig7_partner_heatmap_recount.png`
- 指标：`metric_partner_year_recount.csv`、`metric_top20_partners_recount.csv`
- USA摘要：`usa_recount_summary.txt`

并行说明：
- `--n_jobs 0` = 自动使用 `CPU核数-1`，用于加速大规模地址解析。

## 核心-边缘结构演化（并行优化 + 高质量可视化）

如果你要直接回答“是否由少数核心机构向更均衡网络演进”，可运行：

```bash
python core_periphery_analysis_optimized.py \
  --file1 /data/student2601/WOS/WOS/Funding/merged_paper_info_with_details_full.tsv \
  --file2 /data/student2601/WOS/WOS/Funding/2022merged_paper_info_new_unique.csv \
  --out_dir ./output_core_periphery \
  --n_jobs 0
```

输出：
- `core_periphery_metrics.csv`：按“逐年/分期”的核心-边缘指标（HHI、Gini、CR5、核层占比、LCC占比等）
- `edge_weight_distribution.csv`：各阶段边权分布数据
- `top_institutions_by_time.csv`：各阶段核心机构指标（degree/strength/betweenness）
- `core_periphery_dashboard.png`：4联图总览
- `core_periphery_metrics_heatmap.png`：多指标标准化热力图
- `core_periphery_bubble_tradeoff.png`：密度-集中度权衡气泡图
- `core_periphery_edgeweight_violin.png`：边权分布小提琴图
- `core_periphery_top_institutions_latest.png`：最新阶段Top机构横向条形图
- `core_periphery_top_institutions_stream.png`：Top机构强度堆叠演化图
- `quality_check.txt`：缺失值与过滤说明

如果你已经生成好上述指标，不想再次读取原始大文件，可直接：

```bash
python vis_core_periphery_from_metrics.py \
  --metrics_dir ./output_core_periphery \
  --out_dir ./output_core_periphery/figures_from_metrics
```

This script redraws figures only from `core_periphery_metrics.csv` (plus optional `edge_weight_distribution.csv` and `top_institutions_by_time.csv`) and now uses English labels with Times New Roman as the primary plotting font.

Note: If many years make the x-axis crowded, the new scripts use built-in adaptive tick thinning + 35° rotation + right alignment.

## Gephi 使用说明

详见：`GEPHI_GUIDE.md`


## 从“已重算数据”中排除姓氏噪声并重绘图6/图7

如果你已经跑过 `recount_us_and_redraw_fig6_fig7.py`，可直接用：

```bash
python filter_recount_and_replot_fig6_fig7.py   --metrics_recount_dir /data/student2601/WOS/WOS/0207/output_funding_analysis/metrics_recount   --out_dir /data/student2601/WOS/WOS/0207/output_funding_analysis/figures_recount_filtered
```

输出文件：
- `fig6_top20_partners_recount_filtered.png`
- `fig7_partner_heatmap_recount_filtered.png`
- `metric_partner_year_recount_filtered.csv`
- `metric_top20_partners_recount_filtered.csv`
- `removed_partner_tokens.txt`（被剔除的异常国家token清单）


- Visualization language can be set to full English with Times New Roman-only rendering in the latest scripts.
