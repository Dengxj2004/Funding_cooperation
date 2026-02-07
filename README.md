# Funding Cooperation 分析脚本

该仓库提供 `analysis_pipeline.py`，用于对以下两份大规模数据进行统一清洗、指标计算与图表输出：

- `/data/student2601/WOS/WOS/Funding/merged_paper_info_with_details_full.tsv`
- `/data/student2601/WOS/WOS/Funding/2022merged_paper_info_new_unique.csv`

## 功能覆盖

脚本实现了你提出的核心任务，包括：

- 跨国/跨机构合作数量与占比（总体、逐年）
- 单机构/国内跨机构/国际合作结构
- 合作国家广度
- 按 `grantno` 的平均合作机构数、平均合作作者数
- 中国合作 Top20 国家/地区
- 中国与主要合作伙伴年度合作强度
- 中国第一地址论文趋势
- 中国牵引双边/多边合作及占比
- 按项目类型、按学部的拆解指标
- 图1~图13 对应可视化输出

## 运行环境

建议 Python 3.9+。

安装依赖：

```bash
pip install pandas numpy matplotlib seaborn networkx
```

## 运行方式

```bash
python analysis_pipeline.py \
  --file1 /data/student2601/WOS/WOS/Funding/merged_paper_info_with_details_full.tsv \
  --file2 /data/student2601/WOS/WOS/Funding/2022merged_paper_info_new_unique.csv \
  --out_dir ./output_funding_analysis \
  --simsun /data/student2601/WOS/SimSun.ttf \
  --times "/data/student2601/WOS/Times New Roman.ttf"
```

## 输出目录

- `output_funding_analysis/metrics/`：各指标结果 CSV
- `output_funding_analysis/figures/`：图1~图13 PNG
- `output_funding_analysis/intermediate_*.pkl`：中间表（行级、论文级）

## 关键口径说明

- **国家/地区识别**：地址字段 `C1` 中每个地址最后一个逗号后的字符串视为国家/地区。
- **中国归并规则**：`Hong Kong`、`Macau/Macao`、`Taiwan` 统一归并为 `peoples r china`（小写标准化后处理）。
- **逐年统计起点**：1995 年。
- **五年分段**：
  - 2000年以前
  - 2001-2005
  - 2006-2010
  - 2011-2015
  - 2016-2020
  - 2021-2025
- **机构识别**：地址中第一个逗号前字符串。
- **中国第一地址**：仅看地址列表中的第一个地址是否属于中国。


## 断点续跑（仅补画图8-图13）

如果 `analysis_pipeline.py` 已经生成了：

- `output_funding_analysis/intermediate_paper_level.pkl`
- `output_funding_analysis/metrics/*.csv`

但在网络图阶段因 `scipy` 缺失报错，可直接运行：

```bash
python continue_plots_from_checkpoint.py --out_dir ./output_funding_analysis
```

该脚本会从断点继续绘图（图8~图13），并在 `spring_layout` 不可用时自动降级为不依赖 `scipy` 的布局。
