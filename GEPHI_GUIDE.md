# Gephi 导入与绘图指南（基于本项目导出的点边文件）

本文指导你把 `output_funding_analysis/gephi/` 下的点表和边表导入 Gephi，并快速做出可用网络图。

## 1. 准备文件

推荐先用以下文件：

- 国家网络（总体）
  - `country_nodes_overall.csv`
  - `country_edges_overall.csv`
- 中国-伙伴网络（更聚焦）
  - `china_partner_nodes_overall.csv`
  - `china_partner_edges_overall.csv`
- 机构网络（节点较多，可能较重）
  - `institution_nodes_overall.csv`
  - `institution_edges_overall.csv`

## 2. 导入步骤

1. 打开 Gephi，新建 Project。
2. `File -> Import Spreadsheet`，先导入节点文件（`*_nodes_*.csv`）。
   - As table: `Nodes table`
   - 分隔符：`,`
   - 勾选第一行为字段名
   - `Id` 作为节点ID，`Label` 作为显示名
3. 再导入边文件（`*_edges_*.csv`）。
   - As table: `Edges table`
   - `Source` / `Target` 自动识别
   - `Type` 选 `Undirected`
4. 导入完成后切到 `Overview`。

## 3. 推荐布局（快速出图）

### 方案A（整体关系）
- Layout 选 `ForceAtlas 2`
  - 勾选 `LinLog mode`
  - `Scaling` 2~10（根据图大小调）
  - `Gravity` 1~5
  - 勾选 `Prevent Overlap`
- 运行 30~120 秒，看到节点稳定后停止。

### 方案B（大图防重叠）
- 先 `Fruchterman Reingold` 跑 2000~5000 iterations
- 再切 `ForceAtlas 2` 微调

## 4. 样式映射（重点）

在 `Appearance` 面板：

1. **节点大小**：按 `PaperCount` 映射（Ranking -> Size）
   - Min 10, Max 80（按图密度调）
2. **节点颜色**：按 `IsChina` 或 `Type`
   - `IsChina=1` 设红色/深蓝，其他设浅蓝或绿色
3. **边粗细**：按 `Weight` 映射（Ranking -> Edges -> Size）
   - Min 0.2, Max 8
4. **标签显示**：`Label` 打开
   - Label size 可按节点大小比例
   - 开启 `Label Adjust` 防止重叠

## 5. 过滤与子图

- 只看强连接：在 `Filters -> Edges Weight Range` 中过滤低权重边（例如 >3）。
- 只看核心国家：按节点度数或 `PaperCount` 做 TopN 筛选。
- 分时期网络：导入 `country_nodes_2001_2005.csv` + 对应 `country_edges_2001_2005.csv` 等分别作图。

## 6. 导出高分辨率图片

- 切换 `Preview`
- 调整：
  - Node border/opacity
  - Edge opacity 30%~60%
  - Label font
- `File -> Export -> SVG/PNG/PDF`
  - 论文推荐 SVG 或 PDF（矢量）

## 7. 常见问题

1. **中文乱码**：Gephi 字体改为支持中文的字体（如 SimSun / Microsoft YaHei）。
2. **图太乱**：提高边权阈值、隐藏小节点、先布局后再开标签。
3. **节点太密**：提高 `Scaling` 与 `Gravity`，并启用 `Prevent Overlap`。

