import argparse
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BLUE_GRADIENT = ["#9bc9dd", "#6baed5", "#4392c4"]
GREEN_GRADIENT = ["#a2d59b", "#76c277", "#3bab5a"]


def setup_fonts(simsun: str, times: str):
    if os.path.exists(simsun):
        fm.fontManager.addfont(simsun)
    if os.path.exists(times):
        fm.fontManager.addfont(times)

    # 参考用户给出的稳定中文字体设置
    plt.rcParams["font.sans-serif"] = [
        "SimSun",
        "Times New Roman",
        "SimHei",
        "Microsoft YaHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.facecolor"] = "white"


def style_axis(ax, title: str, xlabel: str, ylabel: str):
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


def load_metric_files(metrics_dir: str):
    metrics_path = os.path.join(metrics_dir, "core_periphery_metrics.csv")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"未找到指标文件: {metrics_path}")

    metrics_df = pd.read_csv(metrics_path)

    edge_path = os.path.join(metrics_dir, "edge_weight_distribution.csv")
    top_path = os.path.join(metrics_dir, "top_institutions_by_time.csv")

    edge_df = pd.read_csv(edge_path) if os.path.exists(edge_path) else pd.DataFrame()
    top_df = pd.read_csv(top_path) if os.path.exists(top_path) else pd.DataFrame()

    time_col = "year" if "year" in metrics_df.columns else "period"
    return metrics_df, edge_df, top_df, time_col


def save(fig, out_dir: str, filename: str):
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_dashboard(metrics_df: pd.DataFrame, time_col: str, out_dir: str):
    x = metrics_df[time_col].astype(str)
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)

    axes[0, 0].plot(x, metrics_df["hhi_degree"], marker="o", linewidth=2.8, color=BLUE_GRADIENT[2], label="HHI(度)")
    axes[0, 0].plot(x, metrics_df["hhi_strength"], marker="s", linewidth=2.2, color=BLUE_GRADIENT[1], label="HHI(强度)")
    style_axis(axes[0, 0], "集中度（HHI）", "时间", "指数")
    axes[0, 0].legend(frameon=False)
    format_time_ticks(axes[0, 0], x, max_ticks=10)

    axes[0, 1].plot(x, metrics_df["gini_degree"], marker="o", linewidth=2.8, color=GREEN_GRADIENT[2], label="Gini(度)")
    axes[0, 1].plot(x, metrics_df["gini_strength"], marker="s", linewidth=2.2, color=GREEN_GRADIENT[1], label="Gini(强度)")
    axes[0, 1].plot(x, metrics_df["cr5_degree"], marker="^", linewidth=2.0, color=BLUE_GRADIENT[0], label="CR5(度)")
    style_axis(axes[0, 1], "不平等与头部占比", "时间", "比例")
    axes[0, 1].legend(frameon=False)
    format_time_ticks(axes[0, 1], x, max_ticks=10)

    axes[1, 0].bar(x, metrics_df["core_ratio"], color=sns.color_palette(BLUE_GRADIENT, len(x)))
    axes[1, 0].plot(x, metrics_df["lcc_ratio"], marker="D", linewidth=2.0, color=GREEN_GRADIENT[2], label="LCC占比")
    style_axis(axes[1, 0], "核心占比与连通性", "时间", "比例")
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
    save(fig, out_dir, "core_periphery_dashboard.png")


def plot_heatmap(metrics_df: pd.DataFrame, time_col: str, out_dir: str):
    cols = ["hhi_degree", "hhi_strength", "gini_degree", "gini_strength", "cr5_degree", "cr5_strength", "core_ratio", "lcc_ratio", "density"]
    hm = metrics_df.set_index(time_col)[cols].copy()
    hm = hm.apply(lambda c: (c - c.mean()) / (c.std() + 1e-12), axis=0)

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(hm.T, cmap=sns.blend_palette([GREEN_GRADIENT[0], "#f8fbff", BLUE_GRADIENT[2]], as_cmap=True), linewidths=0.3, linecolor="white", ax=ax)
    ax.set_title("多指标标准化热力图（按时间）", fontsize=16)
    ax.set_xlabel("时间")
    format_time_ticks(ax, hm.columns.tolist(), max_ticks=10)
    ax.set_ylabel("指标")
    save(fig, out_dir, "core_periphery_metrics_heatmap.png")


def plot_bubble(metrics_df: pd.DataFrame, time_col: str, out_dir: str):
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
    style_axis(ax, "网络密度-集中度权衡图（气泡=节点规模, 颜色=核心占比）", "密度", "HHI(强度)")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("core_ratio")
    save(fig, out_dir, "core_periphery_bubble_tradeoff.png")


def plot_violin(edge_df: pd.DataFrame, time_col: str, out_dir: str):
    if edge_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.violinplot(
        data=edge_df,
        x=time_col,
        y="weight",
        palette=sns.color_palette(BLUE_GRADIENT, n_colors=max(3, edge_df[time_col].nunique())),
        inner="quartile",
        cut=0,
        ax=ax,
    )
    style_axis(ax, "合作强度（边权）分布演化", "时间", "边权（共同出现次数）")
    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_ha("right")
    save(fig, out_dir, "core_periphery_edgeweight_violin.png")


def plot_top_bar(top_df: pd.DataFrame, time_col: str, out_dir: str):
    if top_df.empty:
        return
    latest = top_df[time_col].iloc[-1]
    d = top_df[top_df[time_col] == latest].copy().head(15)
    d = d.sort_values("strength", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(d["institution"], d["strength"], color=sns.color_palette(GREEN_GRADIENT, len(d)))
    style_axis(ax, f"{latest} 核心机构 Top15（按强度）", "加权度（strength）", "机构")
    save(fig, out_dir, "core_periphery_top_institutions_latest.png")


def plot_stream(top_df: pd.DataFrame, time_col: str, out_dir: str):
    if top_df.empty:
        return
    top_global = top_df.groupby("institution")["strength"].sum().sort_values(ascending=False).head(8).index.tolist()
    d = top_df[top_df["institution"].isin(top_global)].copy()
    pv = d.pivot_table(index=time_col, columns="institution", values="strength", aggfunc="sum", fill_value=0)

    fig, ax = plt.subplots(figsize=(13, 7))
    colors = sns.color_palette(BLUE_GRADIENT + GREEN_GRADIENT, n_colors=pv.shape[1])
    ax.stackplot(pv.index.astype(str), [pv[c].values for c in pv.columns], labels=pv.columns, colors=colors, alpha=0.85)
    style_axis(ax, "头部机构合作强度堆叠演化（Top8）", "时间", "合作强度")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_ha("right")
    save(fig, out_dir, "core_periphery_top_institutions_stream.png")


def main(args):
    setup_fonts(args.simsun, args.times)
    sns.set_theme(style="whitegrid", context="talk")

    metrics_df, edge_df, top_df, time_col = load_metric_files(args.metrics_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    plot_dashboard(metrics_df, time_col, args.out_dir)
    plot_heatmap(metrics_df, time_col, args.out_dir)
    plot_bubble(metrics_df, time_col, args.out_dir)
    plot_violin(edge_df, time_col, args.out_dir)
    plot_top_bar(top_df, time_col, args.out_dir)
    plot_stream(top_df, time_col, args.out_dir)

    print("完成：已基于已有指标文件重绘核心-边缘可视化")
    print(f"指标目录: {args.metrics_dir}")
    print(f"输出目录: {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于已生成指标文件重绘核心-边缘图，不重新读取原始数据")
    parser.add_argument("--metrics_dir", default="./output_core_periphery")
    parser.add_argument("--out_dir", default="./output_core_periphery/figures_from_metrics")
    parser.add_argument("--simsun", default="/data/student2601/WOS/SimSun.ttf")
    parser.add_argument("--times", default="/data/student2601/WOS/Times New Roman.ttf")
    args = parser.parse_args()
    main(args)
