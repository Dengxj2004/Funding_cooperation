import argparse
import os
from itertools import combinations
from collections import Counter

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


BLUE_GRADIENT = ["#9bc9dd", "#6baed5", "#4392c4"]
GREEN_GRADIENT = ["#a2d59b", "#76c277", "#3bab5a"]
KEY_APPLY_TYPES = ["面上项目", "重大项目", "重点项目", "青年科学基金项目"]


def setup_fonts(simsun_path: str, times_path: str):
    if os.path.exists(simsun_path):
        fm.fontManager.addfont(simsun_path)
    if os.path.exists(times_path):
        fm.fontManager.addfont(times_path)
    plt.rcParams["font.sans-serif"] = ["SimSun", "Times New Roman", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def period_5y(year):
    y = int(year)
    if y <= 2000:
        return "2000年以前"
    if 2001 <= y <= 2005:
        return "2001-2005"
    if 2006 <= y <= 2010:
        return "2006-2010"
    if 2011 <= y <= 2015:
        return "2011-2015"
    if 2016 <= y <= 2020:
        return "2016-2020"
    if 2021 <= y <= 2025:
        return "2021-2025"
    return "2026+"


def safe_layout(G: nx.Graph, seed: int = 42):
    """优先spring_layout，若缺scipy自动降级，避免中断。"""
    try:
        return nx.spring_layout(G, seed=seed)
    except ModuleNotFoundError:
        if G.number_of_nodes() <= 1:
            return {n: (0.0, 0.0) for n in G.nodes()}
        # 不依赖 scipy 的布局
        return nx.circular_layout(G)


def save_plot(fig, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close(fig)




def _safe_name(x: str):
    import re
    return re.sub(r"[^0-9a-zA-Z一-龥]+", "_", str(x)).strip("_")


def export_gephi_files_from_checkpoint(paper_df: pd.DataFrame, out_dir: str):
    gephi_dir = os.path.join(out_dir, "gephi")
    os.makedirs(gephi_dir, exist_ok=True)

    country_counter = Counter()
    edge_counter = Counter()
    for _, r in paper_df.iterrows():
        countries = sorted(set(r["countries"]))
        for c in countries:
            country_counter[c] += 1
        for a, b in combinations(countries, 2):
            edge_counter[(a, b)] += 1

    pd.DataFrame([
        {"Id": c, "Label": c, "Type": "country", "PaperCount": n, "IsChina": int(c == "peoples r china")}
        for c, n in country_counter.items()
    ]).to_csv(os.path.join(gephi_dir, "country_nodes_overall.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"Source": a, "Target": b, "Weight": w, "Type": "Undirected"}
        for (a, b), w in edge_counter.items()
    ]).to_csv(os.path.join(gephi_dir, "country_edges_overall.csv"), index=False, encoding="utf-8-sig")

    cn_partner = Counter()
    for _, r in paper_df.iterrows():
        cs = set(r["countries"])
        if "peoples r china" in cs:
            for p in cs - {"peoples r china"}:
                cn_partner[p] += 1
    pd.DataFrame([
        {"Id": "peoples r china", "Label": "peoples r china", "Type": "country", "IsChina": 1, "PaperCount": int(sum(cn_partner.values()))}
    ] + [
        {"Id": p, "Label": p, "Type": "country", "IsChina": 0, "PaperCount": int(w)} for p, w in cn_partner.items()
    ]).to_csv(os.path.join(gephi_dir, "china_partner_nodes_overall.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"Source": "peoples r china", "Target": p, "Weight": int(w), "Type": "Undirected"} for p, w in cn_partner.items()
    ]).to_csv(os.path.join(gephi_dir, "china_partner_edges_overall.csv"), index=False, encoding="utf-8-sig")

    for period, g in paper_df.groupby(paper_df["PY"].apply(period_5y)):
        cc = Counter()
        ec = Counter()
        for _, r in g.iterrows():
            countries = sorted(set(r["countries"]))
            for c in countries:
                cc[c] += 1
            for a, b in combinations(countries, 2):
                ec[(a, b)] += 1
        pd.DataFrame([
            {"Id": c, "Label": c, "Type": "country", "PaperCount": n, "IsChina": int(c == "peoples r china"), "Period": period}
            for c, n in cc.items()
        ]).to_csv(os.path.join(gephi_dir, f"country_nodes_{_safe_name(period)}.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame([
            {"Source": a, "Target": b, "Weight": w, "Type": "Undirected", "Period": period}
            for (a, b), w in ec.items()
        ]).to_csv(os.path.join(gephi_dir, f"country_edges_{_safe_name(period)}.csv"), index=False, encoding="utf-8-sig")

def draw_from_checkpoint(paper_df: pd.DataFrame, metrics_dir: str, figure_dir: str):
    os.makedirs(figure_dir, exist_ok=True)

    metric_china_mode_annual = pd.read_csv(os.path.join(metrics_dir, "metric_china_mode_annual.csv"))
    metric_apply_type = pd.read_csv(os.path.join(metrics_dir, "metric_apply_type.csv"))
    metric_fund_org = pd.read_csv(os.path.join(metrics_dir, "metric_fund_org.csv"))

    # 图8
    key_apply = paper_df[paper_df["applyType"].isin(KEY_APPLY_TYPES)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, t in enumerate(KEY_APPLY_TYPES):
        ax = axes.flat[i]
        sub = key_apply[key_apply["applyType"] == t]
        c = Counter()
        for _, r in sub.iterrows():
            s = set(r["countries"])
            if "peoples r china" in s:
                for p in s - {"peoples r china"}:
                    c[p] += 1
        if not c:
            ax.set_title(f"{t}（无数据）")
            ax.axis("off")
            continue
        G = nx.Graph()
        G.add_node("peoples r china")
        for p, w in c.items():
            G.add_edge("peoples r china", p, weight=w)
        pos = safe_layout(G, seed=42)
        max_w = max(c.values()) if c else 1
        sizes = [1200 if n1 == "peoples r china" else 300 + 40 * G.degree(n1) for n1 in G.nodes()]
        widths = [0.5 + d["weight"] / max_w * 4 for _, _, d in G.edges(data=True)]
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=sizes,
            width=widths,
            node_color=[BLUE_GRADIENT[2] if n1 == "peoples r china" else GREEN_GRADIENT[1] for n1 in G.nodes()],
            ax=ax,
            font_size=8,
        )
        ax.set_title(t)
    fig.suptitle("图8 不同项目类型下中国国际合作网络结构图（断点续跑）")
    save_plot(fig, figure_dir, "fig8_applytype_network.png")

    # 图9
    fa = paper_df.groupby("PY")["china_first_addr"].sum().reset_index(name="china_first_addr_papers")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(fa["PY"], fa["china_first_addr_papers"], color=GREEN_GRADIENT[2], marker="o")
    ax.set_title("图9 中国机构为第一地址论文数趋势")
    ax.set_xlabel("年份")
    ax.set_ylabel("论文数")
    save_plot(fig, figure_dir, "fig9_china_first_addr_trend.png")

    # 图10
    cm = metric_china_mode_annual.sort_values("PY")
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()
    ax1.plot(cm["PY"], cm["bilateral"], color=BLUE_GRADIENT[2], label="双边合作论文数")
    ax1.plot(cm["PY"], cm["multilateral"], color=GREEN_GRADIENT[2], label="多边合作论文数")
    ax2.plot(cm["PY"], cm["bilateral_ratio_in_intl"], color=BLUE_GRADIENT[1], linestyle="--", label="双边占比")
    ax2.plot(cm["PY"], cm["multilateral_ratio_in_intl"], color=GREEN_GRADIENT[1], linestyle="--", label="多边占比")
    ax1.set_title("图10 中国牵引双边/多边合作论文数及占比趋势")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left")
    save_plot(fig, figure_dir, "fig10_china_bi_multi_trend.png")

    # 图11
    phase_defs = [
        ("2000年以前", lambda y: y <= 2000),
        ("2001-2010", lambda y: 2001 <= y <= 2010),
        ("2011-2020", lambda y: 2011 <= y <= 2020),
        ("2021-至今", lambda y: y >= 2021),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, (name, cond) in enumerate(phase_defs):
        ax = axes.flat[i]
        sub = paper_df[paper_df["PY"].apply(cond)]
        G = nx.Graph()
        for _, r in sub.iterrows():
            c = [x for x in r["countries"] if x != "peoples r china"]
            for a, b in combinations(sorted(set(c)), 2):
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)
        if G.number_of_nodes() == 0:
            ax.set_title(f"{name}（无数据）")
            ax.axis("off")
            continue
        pos = safe_layout(G, seed=42)
        ew = nx.get_edge_attributes(G, "weight")
        max_w = max(ew.values()) if ew else 1
        widths = [0.4 + d["weight"] / max_w * 3 for _, _, d in G.edges(data=True)]
        nx.draw(G, pos, with_labels=True, node_size=240, width=widths, node_color=BLUE_GRADIENT[1], ax=ax, font_size=7)
        ax.set_title(name)
    fig.suptitle("图11 中国牵引下国际合作网络演化图（去中国节点，断点续跑）")
    save_plot(fig, figure_dir, "fig11_network_evolution.png")

    # 图12
    p = []
    for period in ["2000年以前", "2001-2005", "2006-2010", "2011-2015", "2016-2020", "2021-2025"]:
        sub = paper_df[paper_df["PY"].apply(period_5y) == period]
        G = nx.Graph()
        for _, r in sub.iterrows():
            insts = sorted(set(r["institutions"]))
            for a, b in combinations(insts, 2):
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)
        china_nodes = [n for n in G.nodes() if ("china" in n or "chinese" in n or "acad sci" in n or "univ" in n)]
        if G.number_of_nodes() and china_nodes:
            dc = nx.degree_centrality(G)
            avg_dc = float(np.mean([dc[n] for n in china_nodes]))
            neighbors = set()
            for n in china_nodes:
                neighbors.update(G.neighbors(n))
            connect_n = len(neighbors - set(china_nodes))
        else:
            avg_dc = 0
            connect_n = 0
        p.append({"period5": period, "avg_degree_centrality": avg_dc, "connect_count": connect_n})
    p = pd.DataFrame(p)

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()
    ax1.plot(p["period5"], p["avg_degree_centrality"], color=BLUE_GRADIENT[2], marker="o", label="节点中心性")
    ax2.plot(p["period5"], p["connect_count"], color=GREEN_GRADIENT[2], marker="s", label="连接数量")
    ax1.set_title("图12 中国科研合作网络节点中心性与连接数量趋势")
    ax1.tick_params(axis="x", rotation=30)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left")
    save_plot(fig, figure_dir, "fig12_centrality_connectivity.png")

    # 图13
    ap = metric_apply_type.copy()
    fs = metric_fund_org.copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].bar(ap["applyType"], ap["multilateral_ratio_china_led"], color=BLUE_GRADIENT[1], label="多边合作占比")
    axes[0].set_title("分项目类型")
    axes[0].tick_params(axis="x", rotation=30)
    ax0b = axes[0].twinx()
    ax0b.plot(ap["applyType"], ap["country_breadth"], color=GREEN_GRADIENT[2], marker="o", label="合作国家总数")

    x = np.arange(len(fs))
    w = 0.35
    axes[1].bar(x - w / 2, fs["intl_ratio"], width=w, color=BLUE_GRADIENT[2], label="跨国占比")
    axes[1].bar(x + w / 2, fs["cross_inst_ratio"], width=w, color=GREEN_GRADIENT[2], label="跨机构占比")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(fs["fundOrg"], rotation=30, ha="right")
    axes[1].set_title("分学部")
    axes[1].legend()
    fig.suptitle("图13 不同项目类型/学部合作特征对比")
    save_plot(fig, figure_dir, "fig13_apply_fund_compare.png")


def main(args):
    setup_fonts(args.simsun, args.times)

    output_dir = args.out_dir
    metrics_dir = os.path.join(output_dir, "metrics")
    figure_dir = os.path.join(output_dir, "figures")
    paper_pkl = os.path.join(output_dir, "intermediate_paper_level.pkl")

    if not os.path.exists(paper_pkl):
        raise FileNotFoundError(f"未找到断点文件: {paper_pkl}")
    if not os.path.isdir(metrics_dir):
        raise FileNotFoundError(f"未找到指标目录: {metrics_dir}")

    paper_df = pd.read_pickle(paper_pkl)
    draw_from_checkpoint(paper_df, metrics_dir, figure_dir)
    export_gephi_files_from_checkpoint(paper_df, output_dir)

    print("断点续跑完成：已从中间文件生成图8-图13，并导出Gephi点边文件")
    print(f"图表目录：{figure_dir}")
    print(f"Gephi目录：{os.path.join(output_dir, 'gephi')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 output_funding_analysis 断点继续绘图（图8-图13）")
    parser.add_argument("--out_dir", default="./output_funding_analysis", help="analysis_pipeline.py 已生成的输出目录")
    parser.add_argument("--simsun", default="/data/student2601/WOS/SimSun.ttf")
    parser.add_argument("--times", default="/data/student2601/WOS/Times New Roman.ttf")
    args = parser.parse_args()
    main(args)
