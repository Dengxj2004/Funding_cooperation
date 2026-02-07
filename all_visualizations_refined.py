import argparse
import os
import re

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

COLORS = {
    "light_blue": "#B3E5FC",
    "medium_blue": "#4FC3F7",
    "dark_blue": "#0288D1",
    "light_green": "#C8E6C9",
    "medium_green": "#81C784",
    "dark_green": "#388E3C",
    "light_red": "#F6B3AC",
    "medium_red": "#E57373",
    "light_coral": "#FFAB91",
    "light_purple": "#DAD7EA",
    "light_yellow": "#FFF9C4",
    "gray": "#90A4AE",
}

PERIOD_ORDER = ["2000年以前", "2001-2005", "2006-2010", "2011-2015", "2016-2020", "2021-2025"]
NOISE_TOKENS = {
    "wei", "jing", "yu", "yang", "xin", "jie", "yan", "lei", "hao", "ying", "li", "yi", "wang",
    "zhang", "liu", "chen", "xu", "sun", "wu", "lin", "zhao", "huang", "zhou", "gao", "ma",
}
ALIAS_MAP = {
    "u.s.a": "usa", "u s a": "usa", "us": "usa", "united states": "usa", "united states of america": "usa", "america": "usa",
    "england": "uk", "scotland": "uk", "wales": "uk", "northern ireland": "uk",
    "peoples r china": "china", "people s r china": "china", "peoples republic of china": "china",
}
SHORT_VALID = {"usa", "uk", "uae", "oman", "peru", "mali", "chad", "togo", "laos", "qatar", "niger", "benin"}


def setup_fonts(simsun, times):
    if os.path.exists(simsun):
        fm.fontManager.addfont(simsun)
    if os.path.exists(times):
        fm.fontManager.addfont(times)
    plt.rcParams["font.sans-serif"] = ["SimSun", "Times New Roman", "SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.facecolor"] = "white"


def get_period(y):
    y = int(y)
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
    return "2021-2025"


def normalize_partner(x):
    t = re.sub(r"\s+", " ", str(x).strip().lower().strip(". ;,"))
    return ALIAS_MAP.get(t, t)


def valid_partner(x):
    if not x or x in NOISE_TOKENS:
        return False
    if re.fullmatch(r"[\d\W_]+", x):
        return False
    if len(x) <= 3 and x not in SHORT_VALID:
        return False
    return True


def load_data(metrics_dir):
    annual = pd.read_csv(os.path.join(metrics_dir, "metric_annual_openess.csv"))
    mix = pd.read_csv(os.path.join(metrics_dir, "metric_annual_collab_mix.csv"))
    partner_year = pd.read_csv(os.path.join(metrics_dir, "metric_partner_year.csv"))
    china_mode = pd.read_csv(os.path.join(metrics_dir, "metric_china_mode_annual.csv"))
    apply_df = pd.read_csv(os.path.join(metrics_dir, "metric_apply_type.csv"))
    fund_df = pd.read_csv(os.path.join(metrics_dir, "metric_fund_org.csv"))

    annual["Period"] = annual["PY"].apply(get_period)
    period = annual.groupby("Period", as_index=False).agg(
        Total_Papers=("total_papers", "sum"),
        Cross_Country_Count=("intl_papers", "sum"),
        Cross_Inst_Count=("cross_inst_papers", "sum"),
    )
    period["Cross_Country_Ratio"] = period["Cross_Country_Count"] / period["Total_Papers"]
    period["Cross_Inst_Ratio"] = period["Cross_Inst_Count"] / period["Total_Papers"]
    period["Period"] = pd.Categorical(period["Period"], PERIOD_ORDER, ordered=True)
    period = period.sort_values("Period")

    py = partner_year.copy()
    py["partner"] = py["partner"].apply(normalize_partner)
    py = py[py["partner"].apply(valid_partner)]
    py = py.groupby(["PY", "partner"], as_index=False)["papers"].sum()

    top20 = py.groupby("partner", as_index=False)["papers"].sum().sort_values("papers", ascending=False).head(20)

    return annual, period, mix, py, top20, china_mode, apply_df, fund_df


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_all(annual, period, mix, partner_year, top20, china_mode, apply_df, fund_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 01
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(annual["PY"], annual["intl_ratio"] * 100, color=COLORS["dark_blue"], marker="o", lw=2.5, label="跨国合作比例")
    ax.plot(annual["PY"], annual["cross_inst_ratio"] * 100, color=COLORS["dark_green"], marker="s", lw=2.5, label="跨机构合作比例")
    ax.fill_between(annual["PY"], annual["intl_ratio"] * 100, alpha=0.2, color=COLORS["medium_blue"])
    ax.fill_between(annual["PY"], annual["cross_inst_ratio"] * 100, alpha=0.2, color=COLORS["medium_green"])
    ax.set_title("01 中国NSFC资助论文年度合作趋势")
    ax.set_xlabel("年份"); ax.set_ylabel("比例(%)"); ax.grid(ls="--", alpha=0.4)
    ax.legend()
    save(fig, out_dir, "01_collaboration_trend_line.png")

    # 02
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(period))
    w = 0.28
    ax.bar(x - w, period["Total_Papers"] / 1000, w, color="#9bc9dd", label="论文总数(千)")
    ax.bar(x, period["Cross_Inst_Count"] / 1000, w, color="#6baed5", label="跨机构(千)")
    ax.bar(x + w, period["Cross_Country_Count"] / 1000, w, color="#4392c4", label="跨国(千)")
    ax.set_xticks(x); ax.set_xticklabels(period["Period"], rotation=15)
    ax.set_title("02 五年周期合作统计")
    ax.grid(axis="y", ls="--", alpha=0.4); ax.legend()
    save(fig, out_dir, "02_period_comparison_bar.png")

    # 03
    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(len(period))
    left = period["Cross_Country_Ratio"] * 100
    right = period["Cross_Inst_Ratio"] * 100
    ax.barh(y, -left, 0.35, color=COLORS["light_blue"], edgecolor=COLORS["dark_blue"], label="跨国合作比例")
    ax.barh(y, right, 0.35, color=COLORS["light_green"], edgecolor=COLORS["dark_green"], label="跨机构合作比例")
    ax.axvline(0, color=COLORS["gray"], lw=1.5)
    ax.set_yticks(y); ax.set_yticklabels(period["Period"])
    ax.set_xticklabels([f"{abs(t):.0f}" for t in ax.get_xticks()])
    ax.set_title("03 跨国 vs 跨机构合作比例双向对比")
    ax.grid(axis="x", ls="--", alpha=0.4); ax.legend(loc="upper right")
    save(fig, out_dir, "03_bullet_chart.png")

    # 04
    top20s = top20.sort_values("papers", ascending=True)
    fig, ax = plt.subplots(figsize=(14, 10))
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(top20s)))[::-1]
    ax.barh(top20s["partner"], top20s["papers"], color=colors, edgecolor=COLORS["dark_blue"], lw=0.5)
    ax.set_title("04 中国学术合作Top20伙伴国家/地区")
    ax.set_xlabel("合作论文数量"); ax.grid(axis="x", ls="--", alpha=0.4)
    save(fig, out_dir, "04_top_partners_bar.png")

    # 05
    top15 = partner_year.groupby("partner")["papers"].sum().sort_values(ascending=False).head(15).index
    pv = partner_year[partner_year["partner"].isin(top15)].pivot_table(index="partner", columns="PY", values="papers", aggfunc="sum", fill_value=0)
    pv = pv.loc[pv.sum(axis=1).sort_values(ascending=False).index]
    data_log = np.log10(pv.values + 1)
    cmap = LinearSegmentedColormap.from_list("custom_blues", ['#F7FBFF', '#DEEBF7', '#C6DBEF', '#9ECAE1', '#6BAED6', '#4292C6', '#2171B5', '#08519C', '#08306B'])
    fig, ax = plt.subplots(figsize=(18, 10))
    im = ax.imshow(data_log, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(pv.columns))); ax.set_xticklabels(pv.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(pv.index))); ax.set_yticklabels(pv.index, fontsize=11)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02); cbar.set_label("合作论文数量(log10)")
    ax.set_title("05 中国与主要合作伙伴年度合作热力图")
    save(fig, out_dir, "05_partners_heatmap.png")

    # 06
    m = mix.sort_values("PY").copy()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.stackplot(m["PY"], m["single_inst"], m["domestic_cross_inst"], m["international"],
                 labels=["单一机构", "国内跨机构", "国际合作"], colors=['#a2d59b', '#76c277', '#3bab5a'], alpha=0.85)
    ax.plot(m["PY"], m[["single_inst", "domestic_cross_inst", "international"]].sum(axis=1), color="#2d8a4e", ls="--", marker="o", ms=3)
    ax.set_title("06 合作类型趋势（堆叠面积）")
    ax.set_xlabel("年份"); ax.set_ylabel("论文数量"); ax.grid(ls="--", alpha=0.4); ax.legend()
    save(fig, out_dir, "06_collaboration_area.png")

    # 07
    main9 = partner_year.groupby("partner")["papers"].sum().sort_values(ascending=False).head(9).index.tolist()
    fig, ax = plt.subplots(figsize=(16, 9))
    palette = sns.color_palette("tab10", n_colors=len(main9))
    for i, p in enumerate(main9):
        s = partner_year[partner_year["partner"] == p].groupby("PY")["papers"].sum()
        ax.plot(s.index, s.values, lw=2, marker="o", ms=4, label=p, color=palette[i])
    ax.set_title("07 主要合作伙伴增长趋势")
    ax.set_xlabel("年份"); ax.set_ylabel("合作论文数量"); ax.grid(ls="--", alpha=0.4)
    ax.legend(ncol=3, fontsize=9)
    save(fig, out_dir, "07_partner_growth_trend.png")

    # 08
    fig, axes = plt.subplots(1, 6, figsize=(24, 5))
    colors_pie = [COLORS["light_blue"], COLORS["light_green"], COLORS["light_red"]]
    for ax, (_, r) in zip(axes, period.iterrows()):
        total = r["Total_Papers"]; cc = r["Cross_Country_Count"]; ci_only = r["Cross_Inst_Count"] - cc; single = total - r["Cross_Inst_Count"]
        ax.pie([single, ci_only, cc], labels=None, autopct="%1.1f%%", colors=colors_pie, explode=(0, 0, 0.05), startangle=90, shadow=True, textprops={"fontsize": 9})
        ax.set_title(r["Period"], fontsize=12)
    fig.legend(["单一机构", "国内跨机构", "国际合作"], loc="lower center", ncol=3)
    fig.suptitle("08 各时期合作类型分布")
    save(fig, out_dir, "08_period_pie_charts.png")

    # 09
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(fund_df)); w = 0.35
    ax.bar(x - w/2, fund_df["intl_ratio"] * 100, w, color=COLORS["medium_blue"], label="跨国合作占比")
    ax.bar(x + w/2, fund_df["cross_inst_ratio"] * 100, w, color=COLORS["medium_green"], label="跨机构合作占比")
    ax.set_xticks(x); ax.set_xticklabels(fund_df["fundOrg"], rotation=25, ha="right")
    ax.set_title("09 各学部合作开放性对比")
    ax.set_ylabel("占比(%)"); ax.grid(axis="y", ls="--", alpha=0.4); ax.legend()
    save(fig, out_dir, "09_fund_org_compare.png")

    # 10
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()
    ax1.bar(apply_df["applyType"], apply_df["multilateral_ratio_china_led"] * 100, color=COLORS["light_purple"], edgecolor="#666", label="中国牵引多边占比")
    ax2.plot(apply_df["applyType"], apply_df["country_breadth"], color=COLORS["dark_green"], marker="o", lw=2, label="合作国家总数")
    ax1.tick_params(axis="x", rotation=20)
    ax1.set_ylabel("多边占比(%)"); ax2.set_ylabel("合作国家总数")
    ax1.set_title("10 不同项目类型合作特征")
    save(fig, out_dir, "10_apply_type_compare.png")

    # 11
    cm = china_mode.sort_values("PY")
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax2 = ax1.twinx()
    ax1.plot(cm["PY"], cm["bilateral"], color=COLORS["dark_blue"], marker="o", label="双边论文数")
    ax1.plot(cm["PY"], cm["multilateral"], color=COLORS["dark_green"], marker="s", label="多边论文数")
    ax2.plot(cm["PY"], cm["bilateral_ratio_in_intl"] * 100, color=COLORS["medium_blue"], ls="--", label="双边占比")
    ax2.plot(cm["PY"], cm["multilateral_ratio_in_intl"] * 100, color=COLORS["medium_green"], ls="--", label="多边占比")
    ax1.set_title("11 中国牵引双边/多边合作趋势")
    ax1.set_xlabel("年份")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left")
    ax1.grid(ls="--", alpha=0.35)
    save(fig, out_dir, "11_china_bi_multi_trend.png")


def main(args):
    setup_fonts(args.simsun, args.times)
    annual, period, mix, partner_year, top20, china_mode, apply_df, fund_df = load_data(args.metrics_dir)

    # 导出清洗后的合作伙伴表，便于核查 USA 与异常项
    cleaned_dir = os.path.join(args.out_dir, "cleaned_tables")
    os.makedirs(cleaned_dir, exist_ok=True)
    partner_year.to_csv(os.path.join(cleaned_dir, "partner_year_cleaned.csv"), index=False, encoding="utf-8-sig")
    top20.to_csv(os.path.join(cleaned_dir, "top20_cleaned.csv"), index=False, encoding="utf-8-sig")

    plot_all(annual, period, mix, partner_year, top20, china_mode, apply_df, fund_df, args.out_dir)
    print("完成：已重绘并美化全部图表（11张）")
    print(f"输出目录：{args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于 output_funding_analysis/metrics 全量重绘美化图")
    parser.add_argument("--metrics_dir", default="./output_funding_analysis/metrics")
    parser.add_argument("--out_dir", default="./output_funding_analysis/figures_beautified")
    parser.add_argument("--simsun", default="/data/student2601/WOS/SimSun.ttf")
    parser.add_argument("--times", default="/data/student2601/WOS/Times New Roman.ttf")
    args = parser.parse_args()
    main(args)
