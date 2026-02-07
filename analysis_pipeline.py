import argparse
import os
import re
from collections import Counter, defaultdict
from itertools import combinations

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


# -----------------------------
# 基础配置
# -----------------------------
BLUE_GRADIENT = ["#9bc9dd", "#6baed5", "#4392c4"]
GREEN_GRADIENT = ["#a2d59b", "#76c277", "#3bab5a"]

KEY_APPLY_TYPES = ["面上项目", "重大项目", "重点项目", "青年科学基金项目"]
KEY_FUND_ORGS = [
    "医学科学部",
    "工程与材料科学部",
    "生命科学部",
    "数理科学部",
    "信息科学部",
    "地球科学部",
    "化学科学部",
    "管理科学部",
]


def setup_fonts(simsun_path: str, times_path: str):
    if os.path.exists(simsun_path):
        fm.fontManager.addfont(simsun_path)
    if os.path.exists(times_path):
        fm.fontManager.addfont(times_path)

    plt.rcParams["font.sans-serif"] = ["SimSun", "Times New Roman", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def period_5y(year):
    if pd.isna(year):
        return np.nan
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


def normalize_country(country_raw: str):
    if not country_raw:
        return "unknown"
    c = str(country_raw).strip().lower().strip(". ;")
    c = re.sub(r"\s+", " ", c)

    china_aliases = {
        "peoples r china",
        "people s r china",
        "peoples republic of china",
        "china",
        "hong kong",
        "hong kong sar",
        "hong kong, china",
        "macau",
        "macao",
        "macau, china",
        "macao, china",
        "taiwan",
        "taiwan, china",
    }
    usa_aliases = {
        "usa",
        "u s a",
        "u.s.a",
        "united states",
        "united states of america",
    }
    if c in china_aliases:
        return "peoples r china"
    if c in usa_aliases:
        return "usa"
    return c


def split_addresses(c1: str):
    if pd.isna(c1) or not str(c1).strip():
        return []
    text = str(c1).strip()
    # WOS中地址常由 ; 分隔
    parts = [p.strip() for p in re.split(r";\s*(?=\[|[^\[])", text) if p.strip()]
    if len(parts) == 1 and ";" in text:
        parts = [p.strip() for p in text.split(";") if p.strip()]
    return parts


def parse_country_from_address(addr: str):
    if not addr:
        return "unknown"
    tail = addr.split(",")[-1].strip()
    return normalize_country(tail)


def parse_inst_from_address(addr: str):
    if not addr:
        return "unknown_inst"
    txt = addr.strip()
    if txt.startswith("[") and "]" in txt:
        txt = txt.split("]", 1)[1].strip(" ,")
    inst = txt.split(",")[0].strip().lower()
    inst = re.sub(r"\s+", " ", inst)
    return inst if inst else "unknown_inst"


def parse_authors(row):
    cand = row.get("AF", "") if isinstance(row, dict) else row.AF
    if pd.isna(cand) or not str(cand).strip():
        cand = row.get("AU", "") if isinstance(row, dict) else row.AU
    if pd.isna(cand) or not str(cand).strip():
        return set()
    authors = [a.strip().lower() for a in str(cand).split(";") if a.strip()]
    return set(authors)


def paper_id(row):
    for col in ["UT", "DI", "TI"]:
        val = row.get(col, "") if isinstance(row, dict) else getattr(row, col, "")
        if pd.notna(val) and str(val).strip():
            return str(val).strip().lower()
    return f"fallback_{row.name}"


def read_data(file_path: str):
    df = pd.read_csv(file_path, sep=None, engine="python", dtype=str, low_memory=False)
    needed = [
        "grantno", "PT", "AU", "AF", "TI", "C1", "PY", "UT", "DI",
        "applyType", "fundOrg", "applyYear"
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = ""
    return df[needed].copy()


def preprocess(df: pd.DataFrame):
    df = df.copy()
    df["PY"] = pd.to_numeric(df["PY"], errors="coerce")
    df = df[df["PY"].notna()]
    df = df[df["PY"] >= 1995]
    df["PY"] = df["PY"].astype(int)

    df["paper_id"] = df.apply(paper_id, axis=1)
    df["addresses"] = df["C1"].apply(split_addresses)
    df["countries"] = df["addresses"].apply(lambda x: sorted({parse_country_from_address(a) for a in x if a}))
    df["institutions"] = df["addresses"].apply(lambda x: sorted({parse_inst_from_address(a) for a in x if a}))
    df["authors"] = df.apply(parse_authors, axis=1)

    df["country_n"] = df["countries"].apply(len)
    df["inst_n"] = df["institutions"].apply(len)

    df["is_international"] = df["country_n"] > 1
    df["is_cross_inst"] = df["inst_n"] > 1

    def collab_type(row):
        if row["inst_n"] <= 1:
            return "single_inst"
        if row["country_n"] > 1:
            return "international"
        return "domestic_cross_inst"

    df["collab_type"] = df.apply(collab_type, axis=1)

    def first_addr_country(addresses):
        if not addresses:
            return "unknown"
        return parse_country_from_address(addresses[0])

    df["first_addr_country"] = df["addresses"].apply(first_addr_country)
    df["china_first_addr"] = df["first_addr_country"] == "peoples r china"

    df["period5"] = df["PY"].apply(period_5y)
    return df


def dedup_to_paper_level(df):
    rows = []
    for pid, g in df.groupby("paper_id", sort=False):
        one = g.iloc[0].copy()
        one["grantnos"] = sorted({str(x).strip() for x in g["grantno"].fillna("") if str(x).strip()})
        one["countries"] = sorted(set().union(*g["countries"].tolist())) if len(g) else []
        one["institutions"] = sorted(set().union(*g["institutions"].tolist())) if len(g) else []
        one["authors"] = sorted(set().union(*g["authors"].tolist())) if len(g) else []
        one["country_n"] = len(one["countries"])
        one["inst_n"] = len(one["institutions"])
        one["is_international"] = one["country_n"] > 1
        one["is_cross_inst"] = one["inst_n"] > 1
        if one["inst_n"] <= 1:
            one["collab_type"] = "single_inst"
        elif one["country_n"] > 1:
            one["collab_type"] = "international"
        else:
            one["collab_type"] = "domestic_cross_inst"
        rows.append(one)
    return pd.DataFrame(rows)


def compute_metrics(df_paper: pd.DataFrame, df_row: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # 开放性指标（总体+逐年）
    annual = df_paper.groupby("PY").agg(
        total_papers=("paper_id", "nunique"),
        intl_papers=("is_international", "sum"),
        cross_inst_papers=("is_cross_inst", "sum"),
    ).reset_index()
    annual["intl_ratio"] = annual["intl_papers"] / annual["total_papers"]
    annual["cross_inst_ratio"] = annual["cross_inst_papers"] / annual["total_papers"]

    collab_mix = (
        df_paper.groupby(["PY", "collab_type"]).size().unstack(fill_value=0).reset_index()
    )
    for c in ["single_inst", "domestic_cross_inst", "international"]:
        if c not in collab_mix.columns:
            collab_mix[c] = 0
    collab_mix["total"] = collab_mix[["single_inst", "domestic_cross_inst", "international"]].sum(axis=1)
    for c in ["single_inst", "domestic_cross_inst", "international"]:
        collab_mix[f"{c}_ratio"] = collab_mix[c] / collab_mix["total"]

    breadth_annual = df_paper.groupby("PY")["countries"].apply(
        lambda s: len(set().union(*s.tolist())) if len(s) else 0
    ).reset_index(name="country_breadth")

    # 按grantno的平均合作单位数、作者数
    g_rows = []
    for gno, g in df_row.groupby("grantno"):
        all_inst = set().union(*g["institutions"].tolist()) if len(g) else set()
        all_auth = set().union(*g["authors"].tolist()) if len(g) else set()
        g_rows.append({"grantno": gno, "inst_count": len(all_inst), "author_count": len(all_auth)})
    project_scale = pd.DataFrame(g_rows)
    project_scale_summary = pd.DataFrame([
        {
            "avg_inst_per_grant": project_scale["inst_count"].mean() if len(project_scale) else 0,
            "avg_author_per_grant": project_scale["author_count"].mean() if len(project_scale) else 0,
            "grant_count": len(project_scale),
        }
    ])

    # 中国合作伙伴Top20 + 年度强度
    partner_counter = Counter()
    partner_year = defaultdict(Counter)
    for _, r in df_paper.iterrows():
        cset = set(r["countries"])
        if "peoples r china" in cset and len(cset) > 1:
            partners = sorted(cset - {"peoples r china"})
            for p in partners:
                partner_counter[p] += 1
                partner_year[r["PY"]][p] += 1

    top20 = pd.DataFrame(partner_counter.items(), columns=["partner", "papers"]).sort_values("papers", ascending=False).head(20)
    top10_partners = top20.head(10)["partner"].tolist()

    year_partner_rows = []
    for y, c in partner_year.items():
        for p, v in c.items():
            year_partner_rows.append({"PY": y, "partner": p, "papers": v})
    year_partner_df = pd.DataFrame(year_partner_rows)

    # 中国牵引双边/多边
    china_led = df_paper[df_paper["china_first_addr"]].copy()

    def bilateral_multilateral(countries):
        others = [c for c in countries if c != "peoples r china"]
        if len(others) == 1:
            return "bilateral"
        if len(others) >= 2:
            return "multilateral"
        return "non_international"

    china_led["china_collab_mode"] = china_led["countries"].apply(bilateral_multilateral)
    china_mode_annual = china_led.groupby(["PY", "china_collab_mode"]).size().unstack(fill_value=0).reset_index()
    for c in ["bilateral", "multilateral", "non_international"]:
        if c not in china_mode_annual.columns:
            china_mode_annual[c] = 0
    china_mode_annual["intl_total"] = china_mode_annual["bilateral"] + china_mode_annual["multilateral"]
    china_mode_annual["bilateral_ratio_in_intl"] = np.where(china_mode_annual["intl_total"] > 0, china_mode_annual["bilateral"] / china_mode_annual["intl_total"], 0)
    china_mode_annual["multilateral_ratio_in_intl"] = np.where(china_mode_annual["intl_total"] > 0, china_mode_annual["multilateral"] / china_mode_annual["intl_total"], 0)

    # 分项目类型/学部
    key_apply = df_paper[df_paper["applyType"].isin(KEY_APPLY_TYPES)].copy()
    key_fund = df_paper[df_paper["fundOrg"].isin(KEY_FUND_ORGS)].copy()

    apply_stats = key_apply.groupby("applyType").apply(
        lambda x: pd.Series({
            "paper_n": x["paper_id"].nunique(),
            "intl_ratio": x["is_international"].mean() if len(x) else 0,
            "country_breadth": len(set().union(*x["countries"].tolist())) if len(x) else 0,
            "multilateral_ratio_china_led": (
                x[(x["china_first_addr"]) & (x["is_international"])]
                .assign(mode=lambda d: d["countries"].apply(lambda cs: "multilateral" if len([k for k in cs if k != "peoples r china"]) >= 2 else "bilateral"))
                ["mode"].eq("multilateral").mean()
                if len(x[(x["china_first_addr"]) & (x["is_international"])]) else 0
            )
        })
    ).reset_index()

    fund_stats = key_fund.groupby("fundOrg").apply(
        lambda x: pd.Series({
            "paper_n": x["paper_id"].nunique(),
            "intl_ratio": x["is_international"].mean() if len(x) else 0,
            "cross_inst_ratio": x["is_cross_inst"].mean() if len(x) else 0,
            "country_breadth": len(set().union(*x["countries"].tolist())) if len(x) else 0,
        })
    ).reset_index()

    # 输出CSV
    annual.to_csv(os.path.join(out_dir, "metric_annual_openess.csv"), index=False, encoding="utf-8-sig")
    collab_mix.to_csv(os.path.join(out_dir, "metric_annual_collab_mix.csv"), index=False, encoding="utf-8-sig")
    breadth_annual.to_csv(os.path.join(out_dir, "metric_annual_country_breadth.csv"), index=False, encoding="utf-8-sig")
    project_scale.to_csv(os.path.join(out_dir, "metric_project_scale_by_grant.csv"), index=False, encoding="utf-8-sig")
    project_scale_summary.to_csv(os.path.join(out_dir, "metric_project_scale_summary.csv"), index=False, encoding="utf-8-sig")
    top20.to_csv(os.path.join(out_dir, "metric_top20_partners.csv"), index=False, encoding="utf-8-sig")
    year_partner_df.to_csv(os.path.join(out_dir, "metric_partner_year.csv"), index=False, encoding="utf-8-sig")
    china_mode_annual.to_csv(os.path.join(out_dir, "metric_china_mode_annual.csv"), index=False, encoding="utf-8-sig")
    apply_stats.to_csv(os.path.join(out_dir, "metric_apply_type.csv"), index=False, encoding="utf-8-sig")
    fund_stats.to_csv(os.path.join(out_dir, "metric_fund_org.csv"), index=False, encoding="utf-8-sig")

    return {
        "annual": annual,
        "collab_mix": collab_mix,
        "breadth_annual": breadth_annual,
        "project_scale": project_scale,
        "project_scale_summary": project_scale_summary,
        "top20": top20,
        "year_partner_df": year_partner_df,
        "top10_partners": top10_partners,
        "china_mode_annual": china_mode_annual,
        "apply_stats": apply_stats,
        "fund_stats": fund_stats,
        "china_led": china_led,
    }


def save_plot(fig, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close(fig)


def draw_plots(df_paper, metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    annual = metrics["annual"]

    # 图1
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()
    ax1.plot(annual["PY"], annual["intl_papers"], color=BLUE_GRADIENT[2], label="跨国论文数")
    ax1.plot(annual["PY"], annual["cross_inst_papers"], color=GREEN_GRADIENT[2], label="跨机构论文数")
    ax2.plot(annual["PY"], annual["intl_ratio"], color=BLUE_GRADIENT[1], linestyle="--", label="跨国占比")
    ax2.plot(annual["PY"], annual["cross_inst_ratio"], color=GREEN_GRADIENT[1], linestyle="--", label="跨机构占比")
    ax1.set_title("图1 跨国/跨机构合作论文数及占比趋势")
    ax1.set_xlabel("年份")
    ax1.set_ylabel("论文数")
    ax2.set_ylabel("占比")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left")
    save_plot(fig, out_dir, "fig1_openess_dual_axis.png")

    # 图2
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(annual["PY"], annual["intl_ratio"], color=BLUE_GRADIENT[2], label="跨国占比")
    ax.plot(annual["PY"], annual["cross_inst_ratio"], color=GREEN_GRADIENT[2], label="跨机构占比")
    ax.set_title("图2 年度跨国/跨机构合作比例趋势")
    ax.set_xlabel("年份")
    ax.set_ylabel("占比")
    ax.legend()
    save_plot(fig, out_dir, "fig2_openess_ratio_trend.png")

    # 图3 5年期总体+学部
    tmp = df_paper.copy()
    tmp["period5"] = tmp["PY"].apply(period_5y)
    p = tmp.groupby("period5").agg(intl_ratio=("is_international", "mean"), cross_inst_ratio=("is_cross_inst", "mean")).reset_index()
    p = p[p["period5"].isin(["2000年以前", "2001-2005", "2006-2010", "2011-2015", "2016-2020", "2021-2025"])]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    x = np.arange(len(p))
    w = 0.35
    axes[0].bar(x - w/2, p["intl_ratio"], width=w, color=BLUE_GRADIENT[1], label="跨国占比")
    axes[0].bar(x + w/2, p["cross_inst_ratio"], width=w, color=GREEN_GRADIENT[1], label="跨机构占比")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(p["period5"], rotation=30)
    axes[0].set_title("总体")
    axes[0].legend()

    fs = metrics["fund_stats"]
    axes[1].barh(fs["fundOrg"], fs["intl_ratio"], color=BLUE_GRADIENT[2], alpha=0.7, label="跨国占比")
    axes[1].barh(fs["fundOrg"], fs["cross_inst_ratio"], color=GREEN_GRADIENT[2], alpha=0.5, label="跨机构占比")
    axes[1].set_title("分学部")
    axes[1].legend()
    fig.suptitle("图3 跨国与跨机构合作比例对比")
    save_plot(fig, out_dir, "fig3_overall_fund_compare.png")

    # 图4
    mix = metrics["collab_mix"].sort_values("PY")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.stackplot(
        mix["PY"],
        mix["single_inst_ratio"],
        mix["domestic_cross_inst_ratio"],
        mix["international_ratio"],
        labels=["单机构", "国内跨机构", "国际合作"],
        colors=["#d9d9d9", GREEN_GRADIENT[1], BLUE_GRADIENT[1]],
        alpha=0.9,
    )
    ax.legend(loc="upper left")
    ax.set_title("图4 单机构/国内跨机构/国际合作占比堆叠面积图")
    ax.set_xlabel("年份")
    ax.set_ylabel("占比")
    save_plot(fig, out_dir, "fig4_collab_mix_stack.png")

    # 图5（5年按论文平均合作单位/作者）
    pp = df_paper.copy()
    pp["author_n"] = pp["authors"].apply(len)
    p5 = pp.groupby(pp["PY"].apply(period_5y)).agg(avg_inst=("inst_n", "mean"), avg_author=("author_n", "mean")).reset_index()
    p5 = p5[p5["PY"].isin(["2000年以前", "2001-2005", "2006-2010", "2011-2015", "2016-2020", "2021-2025"])]
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()
    ax1.plot(p5["PY"], p5["avg_inst"], marker="o", color=GREEN_GRADIENT[2], label="平均合作单位数")
    ax2.plot(p5["PY"], p5["avg_author"], marker="s", color=BLUE_GRADIENT[2], label="平均合作作者数")
    ax1.set_title("图5 基金标注论文平均合作规模趋势图")
    ax1.set_xlabel("时间段")
    ax1.set_ylabel("平均合作单位数")
    ax2.set_ylabel("平均合作作者数")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left")
    save_plot(fig, out_dir, "fig5_avg_collab_scale.png")

    # 图6
    top20 = metrics["top20"].sort_values("papers", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top20["partner"], top20["papers"], color=BLUE_GRADIENT[2])
    ax.set_title("图6 中国学术合作Top20国家/地区频次")
    ax.set_xlabel("合作论文频次")
    save_plot(fig, out_dir, "fig6_top20_partners.png")

    # 图7
    yp = metrics["year_partner_df"]
    top10 = metrics["top10_partners"]
    if len(yp) > 0 and top10:
        hm = yp[yp["partner"].isin(top10)].pivot_table(index="partner", columns="PY", values="papers", aggfunc="sum", fill_value=0)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(hm, cmap=sns.light_palette(BLUE_GRADIENT[2], as_cmap=True), ax=ax)
        ax.set_title("图7 中国与主要合作伙伴年度合作强度热力图")
        save_plot(fig, out_dir, "fig7_partner_heatmap.png")

    # 图8 不同项目类型网络
    key_apply = df_paper[df_paper["applyType"].isin(KEY_APPLY_TYPES)]
    n = len(KEY_APPLY_TYPES)
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
        pos = nx.spring_layout(G, seed=42)
        sizes = [1200 if n1 == "peoples r china" else 300 + 40 * G.degree(n1) for n1 in G.nodes()]
        widths = [0.5 + d["weight"] / max(c.values()) * 4 for _, _, d in G.edges(data=True)]
        nx.draw(G, pos, with_labels=True, node_size=sizes, width=widths, node_color=[BLUE_GRADIENT[2] if n1 == "peoples r china" else GREEN_GRADIENT[1] for n1 in G.nodes()], ax=ax, font_size=8)
        ax.set_title(t)
    fig.suptitle("图8 不同项目类型下中国国际合作网络结构图")
    save_plot(fig, out_dir, "fig8_applytype_network.png")

    # 图9
    fa = df_paper.groupby("PY")["china_first_addr"].sum().reset_index(name="china_first_addr_papers")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(fa["PY"], fa["china_first_addr_papers"], color=GREEN_GRADIENT[2], marker="o")
    ax.set_title("图9 中国机构为第一地址论文数趋势")
    ax.set_xlabel("年份")
    ax.set_ylabel("论文数")
    save_plot(fig, out_dir, "fig9_china_first_addr_trend.png")

    # 图10
    cm = metrics["china_mode_annual"].sort_values("PY")
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()
    ax1.plot(cm["PY"], cm["bilateral"], color=BLUE_GRADIENT[2], label="双边合作论文数")
    ax1.plot(cm["PY"], cm["multilateral"], color=GREEN_GRADIENT[2], label="多边合作论文数")
    ax2.plot(cm["PY"], cm["bilateral_ratio_in_intl"], color=BLUE_GRADIENT[1], linestyle="--", label="双边占比")
    ax2.plot(cm["PY"], cm["multilateral_ratio_in_intl"], color=GREEN_GRADIENT[1], linestyle="--", label="多边占比")
    ax1.set_title("图10 中国牵引双边/多边合作论文数及占比趋势")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left")
    save_plot(fig, out_dir, "fig10_china_bi_multi_trend.png")

    # 图11 分阶段国家网络（去中国后）
    phase_defs = [
        ("2000年以前", lambda y: y <= 2000),
        ("2001-2010", lambda y: 2001 <= y <= 2010),
        ("2011-2020", lambda y: 2011 <= y <= 2020),
        ("2021-至今", lambda y: y >= 2021),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, (name, cond) in enumerate(phase_defs):
        ax = axes.flat[i]
        sub = df_paper[df_paper["PY"].apply(cond)]
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
        pos = nx.spring_layout(G, seed=42)
        widths = [0.4 + d["weight"] / max(nx.get_edge_attributes(G, "weight").values()) * 3 for _, _, d in G.edges(data=True)]
        nx.draw(G, pos, with_labels=True, node_size=240, width=widths, node_color=BLUE_GRADIENT[1], ax=ax, font_size=7)
        ax.set_title(name)
    fig.suptitle("图11 中国牵引下国际合作网络演化图（去中国节点）")
    save_plot(fig, out_dir, "fig11_network_evolution.png")

    # 图12 中国机构节点中心性与连接数量（5年）
    p = []
    for period in ["2000年以前", "2001-2005", "2006-2010", "2011-2015", "2016-2020", "2021-2025"]:
        sub = df_paper[df_paper["period5"] == period]
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
    save_plot(fig, out_dir, "fig12_centrality_connectivity.png")

    # 图13 分项目类型+分学部对比
    ap = metrics["apply_stats"]
    fs = metrics["fund_stats"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].bar(ap["applyType"], ap["multilateral_ratio_china_led"], color=BLUE_GRADIENT[1], label="多边合作占比")
    axes[0].set_title("分项目类型")
    axes[0].tick_params(axis="x", rotation=30)
    ax0b = axes[0].twinx()
    ax0b.plot(ap["applyType"], ap["country_breadth"], color=GREEN_GRADIENT[2], marker="o", label="合作国家总数")

    x = np.arange(len(fs))
    w = 0.35
    axes[1].bar(x - w/2, fs["intl_ratio"], width=w, color=BLUE_GRADIENT[2], label="跨国占比")
    axes[1].bar(x + w/2, fs["cross_inst_ratio"], width=w, color=GREEN_GRADIENT[2], label="跨机构占比")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(fs["fundOrg"], rotation=30, ha="right")
    axes[1].set_title("分学部")
    axes[1].legend()
    fig.suptitle("图13 不同项目类型/学部合作特征对比")
    save_plot(fig, out_dir, "fig13_apply_fund_compare.png")



def _safe_name(x: str):
    return re.sub(r"[^0-9a-zA-Z一-龥]+", "_", str(x)).strip("_")


def export_gephi_files(df_paper: pd.DataFrame, out_dir: str):
    gephi_dir = os.path.join(out_dir, "gephi")
    os.makedirs(gephi_dir, exist_ok=True)

    # 1) 国家网络（总体）
    country_counter = Counter()
    edge_counter = Counter()
    for _, r in df_paper.iterrows():
        countries = sorted(set(r["countries"]))
        for c in countries:
            country_counter[c] += 1
        for a, b in combinations(countries, 2):
            edge_counter[(a, b)] += 1

    country_nodes = pd.DataFrame([
        {"Id": c, "Label": c, "Type": "country", "PaperCount": n, "IsChina": int(c == "peoples r china")}
        for c, n in country_counter.items()
    ]).sort_values(["PaperCount", "Id"], ascending=[False, True])
    country_edges = pd.DataFrame([
        {"Source": a, "Target": b, "Weight": w, "Type": "Undirected"}
        for (a, b), w in edge_counter.items()
    ]).sort_values("Weight", ascending=False)

    country_nodes.to_csv(os.path.join(gephi_dir, "country_nodes_overall.csv"), index=False, encoding="utf-8-sig")
    country_edges.to_csv(os.path.join(gephi_dir, "country_edges_overall.csv"), index=False, encoding="utf-8-sig")

    # 2) 中国-伙伴网络（总体）
    cn_partner = Counter()
    for _, r in df_paper.iterrows():
        cs = set(r["countries"])
        if "peoples r china" in cs:
            for p in cs - {"peoples r china"}:
                cn_partner[p] += 1
    cp_nodes = [{"Id": "peoples r china", "Label": "peoples r china", "Type": "country", "IsChina": 1, "PaperCount": int(sum(cn_partner.values()))}]
    cp_nodes += [{"Id": p, "Label": p, "Type": "country", "IsChina": 0, "PaperCount": int(w)} for p, w in cn_partner.items()]
    cp_edges = [{"Source": "peoples r china", "Target": p, "Weight": int(w), "Type": "Undirected"} for p, w in cn_partner.items()]
    pd.DataFrame(cp_nodes).to_csv(os.path.join(gephi_dir, "china_partner_nodes_overall.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(cp_edges).to_csv(os.path.join(gephi_dir, "china_partner_edges_overall.csv"), index=False, encoding="utf-8-sig")

    # 3) 国家网络（分阶段）
    for p, g in df_paper.groupby("period5"):
        c_counter = Counter()
        e_counter = Counter()
        for _, r in g.iterrows():
            countries = sorted(set(r["countries"]))
            for c in countries:
                c_counter[c] += 1
            for a, b in combinations(countries, 2):
                e_counter[(a, b)] += 1
        ndf = pd.DataFrame([
            {"Id": c, "Label": c, "Type": "country", "PaperCount": n, "IsChina": int(c == "peoples r china"), "Period": p}
            for c, n in c_counter.items()
        ])
        edf = pd.DataFrame([
            {"Source": a, "Target": b, "Weight": w, "Type": "Undirected", "Period": p}
            for (a, b), w in e_counter.items()
        ])
        ndf.to_csv(os.path.join(gephi_dir, f"country_nodes_{_safe_name(p)}.csv"), index=False, encoding="utf-8-sig")
        edf.to_csv(os.path.join(gephi_dir, f"country_edges_{_safe_name(p)}.csv"), index=False, encoding="utf-8-sig")

    # 4) 机构网络（总体，可能较大）
    inst_counter = Counter()
    inst_edges = Counter()
    for _, r in df_paper.iterrows():
        insts = sorted(set(r["institutions"]))
        for i in insts:
            inst_counter[i] += 1
        for a, b in combinations(insts, 2):
            inst_edges[(a, b)] += 1
    inst_nodes = pd.DataFrame([
        {"Id": i, "Label": i, "Type": "institution", "PaperCount": n}
        for i, n in inst_counter.items()
    ]).sort_values("PaperCount", ascending=False)
    inst_edges_df = pd.DataFrame([
        {"Source": a, "Target": b, "Weight": w, "Type": "Undirected"}
        for (a, b), w in inst_edges.items()
    ]).sort_values("Weight", ascending=False)
    inst_nodes.to_csv(os.path.join(gephi_dir, "institution_nodes_overall.csv"), index=False, encoding="utf-8-sig")
    inst_edges_df.to_csv(os.path.join(gephi_dir, "institution_edges_overall.csv"), index=False, encoding="utf-8-sig")


def main(args):
    setup_fonts(args.simsun, args.times)

    df1 = read_data(args.file1)
    df2 = read_data(args.file2)
    raw = pd.concat([df1, df2], ignore_index=True)

    prep = preprocess(raw)
    paper_df = dedup_to_paper_level(prep)

    os.makedirs(args.out_dir, exist_ok=True)
    prep.to_pickle(os.path.join(args.out_dir, "intermediate_row_level.pkl"))
    paper_df.to_pickle(os.path.join(args.out_dir, "intermediate_paper_level.pkl"))

    metrics = compute_metrics(paper_df, prep, os.path.join(args.out_dir, "metrics"))
    draw_plots(paper_df, metrics, os.path.join(args.out_dir, "figures"))
    export_gephi_files(paper_df, args.out_dir)

    print("完成：指标、图表与Gephi网络文件已输出")
    print(f"输出目录：{args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基金论文合作网络分析与可视化")
    parser.add_argument("--file1", default="/data/student2601/WOS/WOS/Funding/merged_paper_info_with_details_full.tsv")
    parser.add_argument("--file2", default="/data/student2601/WOS/WOS/Funding/2022merged_paper_info_new_unique.csv")
    parser.add_argument("--out_dir", default="./output_funding_analysis")
    parser.add_argument("--simsun", default="/data/student2601/WOS/SimSun.ttf")
    parser.add_argument("--times", default="/data/student2601/WOS/Times New Roman.ttf")
    args = parser.parse_args()
    main(args)
