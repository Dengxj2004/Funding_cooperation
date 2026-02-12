import argparse
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from multiprocessing import cpu_count

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

BLUE_GRADIENT = ["#9bc9dd", "#6baed5", "#4392c4"]
GREEN_GRADIENT = ["#a2d59b", "#76c277", "#3bab5a"]
PERIOD_ORDER = ["≤2000", "2001-2005", "2006-2010", "2011-2015", "2016-2020", "2021-2025"]


# =============================
# 基础 I/O 与数据清洗
# =============================
def setup_fonts(font_cn: str, font_en: str) -> None:
    if os.path.exists(font_cn):
        fm.fontManager.addfont(font_cn)
    if os.path.exists(font_en):
        fm.fontManager.addfont(font_en)
    plt.rcParams["font.family"] = ["SimSun", "Times New Roman", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def detect_sep(path: str):
    lower = path.lower()
    if lower.endswith(".tsv"):
        return "\t"
    if lower.endswith(".csv"):
        return ","
    return None


def read_single_file(path: str) -> pd.DataFrame:
    sep = detect_sep(path)
    kwargs = {"low_memory": False}
    if sep is None:
        kwargs.update({"sep": None, "engine": "python"})
    else:
        kwargs.update({"sep": sep})

    df = pd.read_csv(path, **kwargs)
    required = ["grantno", "C1", "PY"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"文件 {path} 缺少必要字段: {miss}")

    df = df.loc[:, required].copy()
    df.rename(columns={"C1": "address", "PY": "year"}, inplace=True)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


def assign_period(year: int) -> str:
    if year <= 2000:
        return "≤2000"
    if year <= 2005:
        return "2001-2005"
    if year <= 2010:
        return "2006-2010"
    if year <= 2015:
        return "2011-2015"
    if year <= 2020:
        return "2016-2020"
    return "2021-2025"


def clean_inst_name(text: str) -> str:
    if not text:
        return ""
    inst = re.sub(r"\s+", " ", text.strip())
    inst = re.sub(r"\.$", "", inst)
    return inst if len(inst) > 3 else ""


def extract_institutions_from_c1(c1_text: str) -> tuple:
    if pd.isna(c1_text):
        return tuple()

    insts = set()
    for part in re.split(r";", str(c1_text)):
        no_bracket = re.sub(r"\[.*?\]", "", part).strip()
        if not no_bracket:
            continue
        inst = clean_inst_name(no_bracket.split(",")[0])
        if inst:
            insts.add(inst)
    return tuple(sorted(insts))


def _parse_chunk(values):
    return [extract_institutions_from_c1(v) for v in values]


def parallel_parse_unique_addresses(address_series: pd.Series, n_jobs: int) -> pd.Series:
    uniq = pd.Series(address_series.dropna().unique(), name="address")
    if len(uniq) == 0:
        return pd.Series([tuple()] * len(address_series), index=address_series.index)

    if n_jobs == 1 or len(uniq) < 5000:
        parsed = [extract_institutions_from_c1(v) for v in uniq]
    else:
        chunks = np.array_split(uniq.tolist(), n_jobs)
        chunks = [c.tolist() for c in chunks if len(c) > 0]
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            parsed_chunks = list(ex.map(_parse_chunk, chunks))
        parsed = [x for chunk in parsed_chunks for x in chunk]

    mapper = dict(zip(uniq.tolist(), parsed))
    return address_series.map(mapper).fillna(tuple())


# =============================
# 网络与指标
# =============================
def build_network(inst_lists: pd.Series) -> nx.Graph:
    edge_counter = Counter()
    for insts in inst_lists:
        if len(insts) < 2:
            continue
        for a, b in combinations(insts, 2):
            edge_counter[(a, b)] += 1

    G = nx.Graph()
    for (a, b), w in edge_counter.items():
        G.add_edge(a, b, weight=int(w))
    return G


def gini_coefficient(x: np.ndarray) -> float:
    if len(x) == 0:
        return np.nan
    if np.all(x == 0):
        return 0.0
    xs = np.sort(x)
    n = len(xs)
    return float((2 * np.sum(np.arange(1, n + 1) * xs)) / (n * np.sum(xs)) - (n + 1) / n)


def calculate_metrics(G: nx.Graph) -> dict:
    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "density": np.nan,
            "hhi_degree": np.nan,
            "hhi_strength": np.nan,
            "gini_degree": np.nan,
            "gini_strength": np.nan,
            "cr5_degree": np.nan,
            "cr5_strength": np.nan,
            "core_ratio": np.nan,
            "lcc_ratio": np.nan,
            "assortativity": np.nan,
        }

    deg = np.array([d for _, d in G.degree()], dtype=float)
    strength = np.array([s for _, s in G.degree(weight="weight")], dtype=float)
    total_deg, total_strength = deg.sum(), strength.sum()

    degree_share = deg / total_deg if total_deg > 0 else np.zeros_like(deg)
    strength_share = strength / total_strength if total_strength > 0 else np.zeros_like(strength)

    core = nx.core_number(G) if G.number_of_edges() > 0 else {n: 0 for n in G.nodes}
    max_core = max(core.values()) if core else 0
    core_ratio = sum(1 for _, c in core.items() if c == max_core) / n_nodes

    if G.number_of_edges() > 0:
        lcc_ratio = len(max(nx.connected_components(G), key=len)) / n_nodes
        assortativity = nx.degree_assortativity_coefficient(G)
    else:
        lcc_ratio = 1.0
        assortativity = np.nan

    return {
        "nodes": n_nodes,
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "hhi_degree": float(np.sum(degree_share ** 2)),
        "hhi_strength": float(np.sum(strength_share ** 2)),
        "gini_degree": gini_coefficient(deg),
        "gini_strength": gini_coefficient(strength),
        "cr5_degree": float(np.sort(deg)[-5:].sum() / total_deg) if total_deg > 0 else np.nan,
        "cr5_strength": float(np.sort(strength)[-5:].sum() / total_strength) if total_strength > 0 else np.nan,
        "core_ratio": core_ratio,
        "lcc_ratio": lcc_ratio,
        "assortativity": assortativity,
    }


def pick_time_dimension(df: pd.DataFrame, yearly_min_rows: int):
    year_counts = df.groupby("year").size()
    valid_years = year_counts[year_counts >= yearly_min_rows].index.tolist()
    if len(valid_years) >= 8:
        return "year", sorted(valid_years)
    periods = [p for p in PERIOD_ORDER if p in set(df["period"].dropna())]
    return "period", periods


def summarize_top_institutions(G: nx.Graph, top_n: int = 15):
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["institution", "degree", "strength", "betweenness"]) 
    degree = dict(G.degree())
    strength = dict(G.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(G, k=min(500, max(10, G.number_of_nodes() // 10)), weight="weight", seed=42)
    rows = []
    for n in G.nodes:
        rows.append({
            "institution": n,
            "degree": degree[n],
            "strength": strength[n],
            "betweenness": betweenness[n],
        })
    return pd.DataFrame(rows).sort_values(["strength", "degree"], ascending=False).head(top_n)


def analyze(df: pd.DataFrame, time_col: str, time_values: list):
    metrics_rows = []
    edge_weight_rows = []
    top_inst_frames = []

    for t in time_values:
        group = df[df[time_col] == t]
        G = build_network(group["institutions"])

        m = calculate_metrics(G)
        m[time_col] = t
        m["paper_count"] = len(group)
        metrics_rows.append(m)

        weights = [d["weight"] for _, _, d in G.edges(data=True)]
        if weights:
            for w in weights:
                edge_weight_rows.append({time_col: t, "weight": w})

        top_df = summarize_top_institutions(G, top_n=20)
        if not top_df.empty:
            top_df[time_col] = t
            top_inst_frames.append(top_df)

    metrics_df = pd.DataFrame(metrics_rows)
    cols = [
        time_col,
        "paper_count",
        "nodes",
        "edges",
        "density",
        "hhi_degree",
        "hhi_strength",
        "gini_degree",
        "gini_strength",
        "cr5_degree",
        "cr5_strength",
        "core_ratio",
        "lcc_ratio",
        "assortativity",
    ]
    edge_weight_df = pd.DataFrame(edge_weight_rows)
    top_inst_df = pd.concat(top_inst_frames, ignore_index=True) if top_inst_frames else pd.DataFrame(columns=[time_col, "institution", "degree", "strength", "betweenness"])

    return metrics_df[cols], edge_weight_df, top_inst_df


# =============================
# 可视化（多种形式）
# =============================
def _style_axis(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=16, pad=8)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)


def format_time_ticks(ax, labels, max_ticks: int = 10, rotation: int = 35):
    labels = [str(x) for x in labels]
    n = len(labels)
    if n == 0:
        return
    step = max(1, (n + max_ticks - 1) // max_ticks)
    tick_pos = list(range(0, n, step))
    if tick_pos[-1] != n - 1:
        tick_pos.append(n - 1)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([labels[i] for i in tick_pos], rotation=rotation, ha="right")
    ax.margins(x=0.02)


def plot_dashboard(metrics_df: pd.DataFrame, time_col: str, out_path: str):
    sns.set_theme(style="whitegrid", context="talk")
    x = metrics_df[time_col].astype(str)

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)

    axes[0, 0].plot(x, metrics_df["hhi_degree"], marker="o", linewidth=2.8, color=BLUE_GRADIENT[2], label="HHI(度)")
    axes[0, 0].plot(x, metrics_df["hhi_strength"], marker="s", linewidth=2.2, color=BLUE_GRADIENT[1], label="HHI(强度)")
    _style_axis(axes[0, 0], "集中度（HHI）", "时间", "指数")
    axes[0, 0].legend(frameon=False)
    format_time_ticks(axes[0, 0], x, max_ticks=10)

    axes[0, 1].plot(x, metrics_df["gini_degree"], marker="o", linewidth=2.8, color=GREEN_GRADIENT[2], label="Gini(度)")
    axes[0, 1].plot(x, metrics_df["gini_strength"], marker="s", linewidth=2.2, color=GREEN_GRADIENT[1], label="Gini(强度)")
    axes[0, 1].plot(x, metrics_df["cr5_degree"], marker="^", linewidth=2.0, color=BLUE_GRADIENT[0], label="CR5(度)")
    _style_axis(axes[0, 1], "不平等与头部占比", "时间", "比例")
    axes[0, 1].legend(frameon=False)
    format_time_ticks(axes[0, 1], x, max_ticks=10)

    axes[1, 0].bar(x, metrics_df["core_ratio"], color=sns.color_palette(BLUE_GRADIENT, len(x)))
    axes[1, 0].plot(x, metrics_df["lcc_ratio"], marker="D", linewidth=2.0, color=GREEN_GRADIENT[2], label="LCC占比")
    _style_axis(axes[1, 0], "核心占比与连通性", "时间", "比例")
    axes[1, 0].legend(frameon=False)
    format_time_ticks(axes[1, 0], x, max_ticks=10)

    axes[1, 1].plot(x, metrics_df["nodes"], marker="o", linewidth=2.8, color=BLUE_GRADIENT[2], label="节点数")
    axes[1, 1].plot(x, metrics_df["edges"], marker="s", linewidth=2.8, color=GREEN_GRADIENT[2], label="边数")
    ax2 = axes[1, 1].twinx()
    ax2.plot(x, metrics_df["density"], marker="^", linewidth=2.0, color=BLUE_GRADIENT[0], linestyle="--", label="密度")
    axes[1, 1].set_title("网络规模与密度", fontsize=16, pad=10)
    axes[1, 1].set_xlabel("时间", fontsize=12)
    axes[1, 1].set_ylabel("规模", fontsize=12)
    ax2.set_ylabel("密度", fontsize=12)
    axes[1, 1].grid(alpha=0.25, linestyle="--", linewidth=0.8)
    format_time_ticks(axes[1, 1], x, max_ticks=10)
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")

    fig.suptitle("国家自然科学基金论文机构合作网络：核心-边缘结构演化", fontsize=20, y=1.02)
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_metric_heatmap(metrics_df: pd.DataFrame, time_col: str, out_path: str):
    mat_cols = ["hhi_degree", "hhi_strength", "gini_degree", "gini_strength", "cr5_degree", "cr5_strength", "core_ratio", "lcc_ratio", "density"]
    hm = metrics_df.set_index(time_col)[mat_cols].copy()
    hm = hm.apply(lambda c: (c - c.mean()) / (c.std() + 1e-12), axis=0)

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(hm.T, cmap=sns.blend_palette([GREEN_GRADIENT[0], "#f8fbff", BLUE_GRADIENT[2]], as_cmap=True), linewidths=0.3, linecolor="white", ax=ax)
    ax.set_title("多指标标准化热力图（按时间）", fontsize=16)
    ax.set_xlabel("时间")
    format_time_ticks(ax, hm.columns.tolist(), max_ticks=10)
    ax.set_ylabel("指标")
    fig.tight_layout()
    fig.savefig(out_path, dpi=350)
    plt.close(fig)


def plot_bubble_tradeoff(metrics_df: pd.DataFrame, time_col: str, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    size = (metrics_df["nodes"].fillna(0) + 1) * 18
    sc = ax.scatter(
        metrics_df["density"],
        metrics_df["hhi_strength"],
        s=size,
        c=metrics_df["core_ratio"],
        cmap=sns.light_palette(BLUE_GRADIENT[2], as_cmap=True),
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
    )
    for _, r in metrics_df.iterrows():
        ax.text(r["density"], r["hhi_strength"], str(r[time_col]), fontsize=9, ha="left", va="bottom")
    _style_axis(ax, "网络密度-集中度权衡图（气泡=节点规模, 颜色=核心占比）", "密度", "HHI(强度)")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("core_ratio")
    fig.tight_layout()
    fig.savefig(out_path, dpi=350)
    plt.close(fig)


def plot_edge_weight_violin(edge_weight_df: pd.DataFrame, time_col: str, out_path: str):
    if edge_weight_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.violinplot(data=edge_weight_df, x=time_col, y="weight", palette=sns.color_palette(BLUE_GRADIENT, n_colors=max(3, edge_weight_df[time_col].nunique())), inner="quartile", cut=0, ax=ax)
    _style_axis(ax, "合作强度（边权）分布演化", "时间", "边权（共同出现次数）")
    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=350)
    plt.close(fig)


def plot_top_institutions_bar(top_inst_df: pd.DataFrame, time_col: str, out_path: str):
    if top_inst_df.empty:
        return
    latest = top_inst_df[time_col].iloc[-1]
    d = top_inst_df[top_inst_df[time_col] == latest].copy().head(15)
    d = d.sort_values("strength", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(d["institution"], d["strength"], color=sns.color_palette(GREEN_GRADIENT, len(d)))
    _style_axis(ax, f"{latest} 核心机构 Top15（按强度）", "加权度（strength）", "机构")
    fig.tight_layout()
    fig.savefig(out_path, dpi=350)
    plt.close(fig)


def plot_top_institution_stream(top_inst_df: pd.DataFrame, time_col: str, out_path: str):
    if top_inst_df.empty:
        return
    top_global = top_inst_df.groupby("institution")["strength"].sum().sort_values(ascending=False).head(8).index.tolist()
    d = top_inst_df[top_inst_df["institution"].isin(top_global)].copy()
    pv = d.pivot_table(index=time_col, columns="institution", values="strength", aggfunc="sum", fill_value=0)

    fig, ax = plt.subplots(figsize=(13, 7))
    colors = sns.color_palette(BLUE_GRADIENT + GREEN_GRADIENT, n_colors=pv.shape[1])
    ax.stackplot(pv.index.astype(str), [pv[c].values for c in pv.columns], labels=pv.columns, colors=colors, alpha=0.85)
    _style_axis(ax, "头部机构合作强度堆叠演化（Top8）", "时间", "合作强度")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


# =============================
# 质量检查 + 主程序
# =============================
def quality_check(df_before_filter: pd.DataFrame, df_after_filter: pd.DataFrame, metrics_df: pd.DataFrame) -> str:
    null_ratio = df_before_filter[["grantno", "address", "year"]].isna().mean().round(4)
    bad_inst = (df_before_filter["institutions"].str.len() < 2).mean().round(4)
    msg = [
        "=== 数据质量检查 ===",
        f"缺失比例 grantno/address/year: {null_ratio.to_dict()}",
        f"机构数<2 的样本占比(过滤前): {bad_inst}",
        f"分析阶段数: {len(metrics_df)}",
        f"原始样本量: {len(df_before_filter)}, 过滤后样本量: {len(df_after_filter)}",
    ]
    if metrics_df[["hhi_degree", "gini_degree", "cr5_degree"]].isna().any().any():
        msg.append("警告: 部分阶段网络过稀导致集中度指标为 NaN（已保留，避免误导）。")
    return "\n".join(msg)


def run(file1: str, file2: str, out_dir: str, n_jobs: int, yearly_min_rows: int, font_cn: str, font_en: str):
    setup_fonts(font_cn, font_en)
    os.makedirs(out_dir, exist_ok=True)

    jobs = max(1, cpu_count() if n_jobs == 0 else n_jobs)
    df1 = read_single_file(file1)
    df2 = read_single_file(file2)
    df = pd.concat([df1, df2], ignore_index=True)

    df = df.dropna(subset=["grantno", "address", "year"]).copy()
    df["year"] = df["year"].astype(int)
    df["period"] = df["year"].apply(assign_period)
    df["institutions"] = parallel_parse_unique_addresses(df["address"], n_jobs=jobs)

    before_filter = df.copy()
    df = df[df["institutions"].str.len() >= 2].copy()

    time_col, time_values = pick_time_dimension(df, yearly_min_rows)
    metrics_df, edge_weight_df, top_inst_df = analyze(df, time_col, time_values)

    # 输出表
    metrics_df.to_csv(os.path.join(out_dir, "core_periphery_metrics.csv"), index=False, encoding="utf-8-sig")
    edge_weight_df.to_csv(os.path.join(out_dir, "edge_weight_distribution.csv"), index=False, encoding="utf-8-sig")
    top_inst_df.to_csv(os.path.join(out_dir, "top_institutions_by_time.csv"), index=False, encoding="utf-8-sig")

    # 多图输出
    plot_dashboard(metrics_df, time_col, os.path.join(out_dir, "core_periphery_dashboard.png"))
    plot_metric_heatmap(metrics_df, time_col, os.path.join(out_dir, "core_periphery_metrics_heatmap.png"))
    plot_bubble_tradeoff(metrics_df, time_col, os.path.join(out_dir, "core_periphery_bubble_tradeoff.png"))
    plot_edge_weight_violin(edge_weight_df, time_col, os.path.join(out_dir, "core_periphery_edgeweight_violin.png"))
    plot_top_institutions_bar(top_inst_df, time_col, os.path.join(out_dir, "core_periphery_top_institutions_latest.png"))
    plot_top_institution_stream(top_inst_df, time_col, os.path.join(out_dir, "core_periphery_top_institutions_stream.png"))

    qc = quality_check(before_filter, df, metrics_df)
    with open(os.path.join(out_dir, "quality_check.txt"), "w", encoding="utf-8") as f:
        f.write(qc + "\n")

    print(qc)
    print(f"时间粒度: {time_col}, 阶段: {time_values}")
    print(f"结果目录: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基金论文机构合作网络核心-边缘演化分析（并行优化+多可视化输出）")
    parser.add_argument("--file1", required=True, help="22年以前 TSV 文件")
    parser.add_argument("--file2", required=True, help="22-23年 CSV 文件")
    parser.add_argument("--out_dir", default="./output_core_periphery")
    parser.add_argument("--n_jobs", type=int, default=0, help="并行进程数，0表示自动使用全部CPU")
    parser.add_argument("--yearly_min_rows", type=int, default=200, help="逐年分析最小样本阈值")
    parser.add_argument("--font_cn", default="/data/student2601/WOS/SimSun.ttf")
    parser.add_argument("--font_en", default="/data/student2601/WOS/Times New Roman.ttf")
    args = parser.parse_args()

    run(
        file1=args.file1,
        file2=args.file2,
        out_dir=args.out_dir,
        n_jobs=args.n_jobs,
        yearly_min_rows=args.yearly_min_rows,
        font_cn=args.font_cn,
        font_en=args.font_en,
    )
