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
