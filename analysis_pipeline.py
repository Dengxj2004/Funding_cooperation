import argparse
import os
import re
from collections import Counter, defaultdict
from itertools import combinations
from multiprocessing import Pool, cpu_count

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

BLUE_GRADIENT = ["#9bc9dd", "#6baed5", "#4392c4"]
GREEN_GRADIENT = ["#a2d59b", "#76c277", "#3bab5a"]
KEY_APPLY_TYPES = ["面上项目", "重大项目", "重点项目", "青年科学基金项目"]
KEY_FUND_ORGS = ["医学科学部", "工程与材料科学部", "生命科学部", "数理科学部", "信息科学部", "地球科学部", "化学科学部", "管理科学部"]


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
    if y <= 2005:
        return "2001-2005"
    if y <= 2010:
        return "2006-2010"
    if y <= 2015:
        return "2011-2015"
    if y <= 2020:
        return "2016-2020"
    if y <= 2025:
        return "2021-2025"
    return "2026+"


def normalize_country(country_raw: str):
    if not country_raw:
        return "unknown"
    c = re.sub(r"\s+", " ", str(country_raw).strip().lower().strip(". ;"))
    china_aliases = {
        "peoples r china", "people s r china", "peoples republic of china", "china",
        "hong kong", "hong kong sar", "hong kong, china",
        "macau", "macao", "macau, china", "macao, china",
        "taiwan", "taiwan, china",
    }
    usa_aliases = {"usa", "u s a", "u.s.a", "united states", "united states of america", "us"}
    if c in china_aliases:
        return "peoples r china"
    if c in usa_aliases:
        return "usa"
        return "united states"
    return c


def split_addresses(c1: str):
    if pd.isna(c1) or not str(c1).strip():
        return []
    t = str(c1).strip()
    parts = [p.strip() for p in t.split(";") if p.strip()]
    return parts


def parse_country_from_address(addr: str):
    if not addr:
        return "unknown"
    return normalize_country(addr.split(",")[-1].strip())


def parse_inst_from_address(addr: str):
    if not addr:
        return "unknown_inst"
    txt = addr.strip()
    if txt.startswith("[") and "]" in txt:
        txt = txt.split("]", 1)[1].strip(" ,")
    inst = re.sub(r"\s+", " ", txt.split(",")[0].strip().lower())
    return inst if inst else "unknown_inst"


def parse_row_payload(payload):
    c1, af, au = payload
    addresses = split_addresses(c1)
    countries = sorted({parse_country_from_address(a) for a in addresses if a})
    insts = sorted({parse_inst_from_address(a) for a in addresses if a})
    a_src = af if (isinstance(af, str) and af.strip()) else au
    authors = sorted({x.strip().lower() for x in str(a_src).split(";") if str(x).strip()}) if isinstance(a_src, str) else []
    first_country = parse_country_from_address(addresses[0]) if addresses else "unknown"
    return countries, insts, authors, first_country


def parse_parallel(df: pd.DataFrame, n_jobs: int):
    payloads = list(zip(df["C1"].fillna(""), df["AF"].fillna(""), df["AU"].fillna("")))
    if n_jobs <= 1:
        result = [parse_row_payload(p) for p in payloads]
    else:
        with Pool(processes=n_jobs) as pool:
            result = list(pool.imap(parse_row_payload, payloads, chunksize=2000))
    countries, insts, authors, first_country = zip(*result) if result else ([], [], [], [])
    df["countries"] = list(countries)
    df["institutions"] = list(insts)
    df["authors"] = list(authors)
    df["first_addr_country"] = list(first_country)
    return df


def read_data(file_path: str):
    df = pd.read_csv(file_path, sep=None, engine="python", dtype=str, low_memory=False)
    needed = ["grantno", "PT", "AU", "AF", "TI", "C1", "PY", "UT", "DI", "applyType", "fundOrg", "applyYear"]
    for c in needed:
        if c not in df.columns:
            df[c] = ""
    return df[needed].copy()


def preprocess(df: pd.DataFrame, n_jobs: int):
    df = df.copy()
    df["PY"] = pd.to_numeric(df["PY"], errors="coerce")
    df = df[df["PY"].notna() & (df["PY"] >= 1995)]
    df["PY"] = df["PY"].astype(int)

    # fast paper id precedence: UT > DI > TI > fallback
    ut = df["UT"].fillna("").str.strip().str.lower()
    di = df["DI"].fillna("").str.strip().str.lower()
    ti = df["TI"].fillna("").str.strip().str.lower()
    fallback = "fallback_" + df.index.astype(str)
    df["paper_id"] = np.where(ut != "", ut, np.where(di != "", di, np.where(ti != "", ti, fallback)))

    df = parse_parallel(df, n_jobs=n_jobs)
    df["country_n"] = df["countries"].str.len()
    df["inst_n"] = df["institutions"].str.len()
    df["is_international"] = df["country_n"] > 1
    df["is_cross_inst"] = df["inst_n"] > 1
    df["collab_type"] = np.where(df["inst_n"] <= 1, "single_inst", np.where(df["country_n"] > 1, "international", "domestic_cross_inst"))
    df["china_first_addr"] = df["first_addr_country"].eq("peoples r china")
    df["period5"] = df["PY"].apply(period_5y)
    return df


def dedup_to_paper_level(df):
    rows = []
    for _, g in df.groupby("paper_id", sort=False):
        one = g.iloc[0].copy()
        one["countries"] = sorted(set().union(*g["countries"].tolist()))
        one["institutions"] = sorted(set().union(*g["institutions"].tolist()))
        one["authors"] = sorted(set().union(*g["authors"].tolist()))
        one["country_n"] = len(one["countries"])
        one["inst_n"] = len(one["institutions"])
        one["is_international"] = one["country_n"] > 1
        one["is_cross_inst"] = one["inst_n"] > 1
        one["collab_type"] = "single_inst" if one["inst_n"] <= 1 else ("international" if one["country_n"] > 1 else "domestic_cross_inst")
        rows.append(one)
    return pd.DataFrame(rows)


def compute_metrics(df_paper: pd.DataFrame, df_row: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    annual = df_paper.groupby("PY").agg(total_papers=("paper_id", "nunique"), intl_papers=("is_international", "sum"), cross_inst_papers=("is_cross_inst", "sum")).reset_index()
    annual["intl_ratio"] = annual["intl_papers"] / annual["total_papers"]
    annual["cross_inst_ratio"] = annual["cross_inst_papers"] / annual["total_papers"]

    collab_mix = df_paper.groupby(["PY", "collab_type"]).size().unstack(fill_value=0).reset_index()
    for c in ["single_inst", "domestic_cross_inst", "international"]:
        if c not in collab_mix.columns:
            collab_mix[c] = 0
    collab_mix["total"] = collab_mix[["single_inst", "domestic_cross_inst", "international"]].sum(axis=1)
    for c in ["single_inst", "domestic_cross_inst", "international"]:
        collab_mix[f"{c}_ratio"] = collab_mix[c] / collab_mix["total"]

    breadth_annual = df_paper.groupby("PY")["countries"].apply(lambda s: len(set().union(*s.tolist())) if len(s) else 0).reset_index(name="country_breadth")

    g_rows = []
    for gno, g in df_row.groupby("grantno"):
        g_rows.append({"grantno": gno, "inst_count": len(set().union(*g["institutions"].tolist())), "author_count": len(set().union(*g["authors"].tolist()))})
    project_scale = pd.DataFrame(g_rows)
    project_scale_summary = pd.DataFrame([{"avg_inst_per_grant": project_scale["inst_count"].mean() if len(project_scale) else 0, "avg_author_per_grant": project_scale["author_count"].mean() if len(project_scale) else 0, "grant_count": len(project_scale)}])

    partner_counter, partner_year = Counter(), defaultdict(Counter)
    for _, r in df_paper.iterrows():
        cset = set(r["countries"])
        if "peoples r china" in cset and len(cset) > 1:
            for p in (cset - {"peoples r china"}):
                partner_counter[p] += 1
                partner_year[r["PY"]][p] += 1
    top20 = pd.DataFrame(partner_counter.items(), columns=["partner", "papers"]).sort_values("papers", ascending=False).head(20)
    year_partner_df = pd.DataFrame([{"PY": y, "partner": p, "papers": v} for y, c in partner_year.items() for p, v in c.items()])

    china_led = df_paper[df_paper["china_first_addr"]].copy()
    def mode(cs):
        n = len([x for x in cs if x != "peoples r china"])
        return "bilateral" if n == 1 else ("multilateral" if n >= 2 else "non_international")
    china_led["china_collab_mode"] = china_led["countries"].apply(mode)
    china_mode_annual = china_led.groupby(["PY", "china_collab_mode"]).size().unstack(fill_value=0).reset_index()
    for c in ["bilateral", "multilateral", "non_international"]:
        if c not in china_mode_annual.columns:
            china_mode_annual[c] = 0
    china_mode_annual["intl_total"] = china_mode_annual["bilateral"] + china_mode_annual["multilateral"]
    china_mode_annual["bilateral_ratio_in_intl"] = np.where(china_mode_annual["intl_total"] > 0, china_mode_annual["bilateral"] / china_mode_annual["intl_total"], 0)
    china_mode_annual["multilateral_ratio_in_intl"] = np.where(china_mode_annual["intl_total"] > 0, china_mode_annual["multilateral"] / china_mode_annual["intl_total"], 0)

    key_apply = df_paper[df_paper["applyType"].isin(KEY_APPLY_TYPES)]
    key_fund = df_paper[df_paper["fundOrg"].isin(KEY_FUND_ORGS)]

    apply_stats = key_apply.groupby("applyType").apply(lambda x: pd.Series({
        "paper_n": x["paper_id"].nunique(),
        "intl_ratio": x["is_international"].mean() if len(x) else 0,
        "country_breadth": len(set().union(*x["countries"].tolist())) if len(x) else 0,
        "multilateral_ratio_china_led": (
            x[(x["china_first_addr"]) & (x["is_international"])]["countries"]
            .apply(lambda cs: len([k for k in cs if k != "peoples r china"]) >= 2).mean()
            if len(x[(x["china_first_addr"]) & (x["is_international"])]) else 0
        )
    })).reset_index()

    fund_stats = key_fund.groupby("fundOrg").apply(lambda x: pd.Series({
        "paper_n": x["paper_id"].nunique(),
        "intl_ratio": x["is_international"].mean() if len(x) else 0,
        "cross_inst_ratio": x["is_cross_inst"].mean() if len(x) else 0,
        "country_breadth": len(set().union(*x["countries"].tolist())) if len(x) else 0,
    })).reset_index()

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

    return {"annual": annual, "collab_mix": collab_mix, "top20": top20, "year_partner_df": year_partner_df, "china_mode_annual": china_mode_annual, "apply_stats": apply_stats, "fund_stats": fund_stats}


def save_plot(fig, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close(fig)


def draw_plots(df_paper, metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    annual = metrics["annual"].sort_values("PY")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(annual["PY"], annual["intl_ratio"], label="跨国占比", color=BLUE_GRADIENT[2])
    ax.plot(annual["PY"], annual["cross_inst_ratio"], label="跨机构占比", color=GREEN_GRADIENT[2])
    ax.legend(); ax.set_title("图2 年度跨国/跨机构合作比例趋势")
    save_plot(fig, out_dir, "fig2_openess_ratio_trend.png")

    top20 = metrics["top20"].sort_values("papers", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top20["partner"], top20["papers"], color=BLUE_GRADIENT[2])
    ax.set_title("图6 中国学术合作Top20国家/地区频次")
    save_plot(fig, out_dir, "fig6_top20_partners.png")

    yp = metrics["year_partner_df"]
    if len(yp):
        top10 = top20.sort_values("papers", ascending=False).head(10)["partner"].tolist()
        hm = yp[yp["partner"].isin(top10)].pivot_table(index="partner", columns="PY", values="papers", aggfunc="sum", fill_value=0)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(hm, cmap=sns.light_palette(BLUE_GRADIENT[2], as_cmap=True), ax=ax)
        ax.set_title("图7 中国与主要合作伙伴年度合作强度热力图")
        save_plot(fig, out_dir, "fig7_partner_heatmap.png")


def _safe_name(x: str):
    return re.sub(r"[^0-9a-zA-Z一-龥]+", "_", str(x)).strip("_")


def export_gephi_files(df_paper: pd.DataFrame, out_dir: str):
    gephi_dir = os.path.join(out_dir, "gephi")
    os.makedirs(gephi_dir, exist_ok=True)
    country_counter, edge_counter = Counter(), Counter()
    inst_counter, inst_edges = Counter(), Counter()

    for _, r in df_paper.iterrows():
        countries = sorted(set(r["countries"]))
        insts = sorted(set(r["institutions"]))
        for c in countries:
            country_counter[c] += 1
        for i in insts:
            inst_counter[i] += 1
        for a, b in combinations(countries, 2):
            edge_counter[(a, b)] += 1
        for a, b in combinations(insts, 2):
            inst_edges[(a, b)] += 1

    pd.DataFrame([{"Id": c, "Label": c, "Type": "country", "PaperCount": n, "IsChina": int(c == "peoples r china")} for c, n in country_counter.items()]).to_csv(os.path.join(gephi_dir, "country_nodes_overall.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([{"Source": a, "Target": b, "Weight": w, "Type": "Undirected"} for (a, b), w in edge_counter.items()]).to_csv(os.path.join(gephi_dir, "country_edges_overall.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([{"Id": i, "Label": i, "Type": "institution", "PaperCount": n} for i, n in inst_counter.items()]).to_csv(os.path.join(gephi_dir, "institution_nodes_overall.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([{"Source": a, "Target": b, "Weight": w, "Type": "Undirected"} for (a, b), w in inst_edges.items()]).to_csv(os.path.join(gephi_dir, "institution_edges_overall.csv"), index=False, encoding="utf-8-sig")

    for p, g in df_paper.groupby("period5"):
        cc, ec = Counter(), Counter()
        for _, r in g.iterrows():
            countries = sorted(set(r["countries"]))
            for c in countries:
                cc[c] += 1
            for a, b in combinations(countries, 2):
                ec[(a, b)] += 1
        pd.DataFrame([{"Id": c, "Label": c, "Type": "country", "PaperCount": n, "Period": p, "IsChina": int(c == "peoples r china")} for c, n in cc.items()]).to_csv(os.path.join(gephi_dir, f"country_nodes_{_safe_name(p)}.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame([{"Source": a, "Target": b, "Weight": w, "Type": "Undirected", "Period": p} for (a, b), w in ec.items()]).to_csv(os.path.join(gephi_dir, f"country_edges_{_safe_name(p)}.csv"), index=False, encoding="utf-8-sig")


def main(args):
    setup_fonts(args.simsun, args.times)
    n_jobs = args.n_jobs if args.n_jobs > 0 else max(1, cpu_count() - 1)

    df1 = read_data(args.file1)
    df2 = read_data(args.file2)
    raw = pd.concat([df1, df2], ignore_index=True)

    prep = preprocess(raw, n_jobs=n_jobs)
    paper_df = dedup_to_paper_level(prep)

    os.makedirs(args.out_dir, exist_ok=True)
    metrics = compute_metrics(paper_df, prep, os.path.join(args.out_dir, "metrics"))
    draw_plots(paper_df, metrics, os.path.join(args.out_dir, "figures"))
    export_gephi_files(paper_df, args.out_dir)

    print("完成：已重新全量运行（无断点），输出 metrics / figures / gephi")
    print(f"n_jobs={n_jobs}, 输出目录={args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基金论文合作分析（全量重跑，多核加速）")
    parser.add_argument("--file1", default="/data/student2601/WOS/WOS/Funding/merged_paper_info_with_details_full.tsv")
    parser.add_argument("--file2", default="/data/student2601/WOS/WOS/Funding/2022merged_paper_info_new_unique.csv")
    parser.add_argument("--out_dir", default="./output_funding_analysis")
    parser.add_argument("--simsun", default="/data/student2601/WOS/SimSun.ttf")
    parser.add_argument("--times", default="/data/student2601/WOS/Times New Roman.ttf")
    parser.add_argument("--n_jobs", type=int, default=0, help="并行进程数，0表示自动使用cpu_count()-1")
    args = parser.parse_args()
    main(args)
