import argparse
import os
import re

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BLUE_GRADIENT = ["#9bc9dd", "#6baed5", "#4392c4"]
GREEN_GRADIENT = ["#a2d59b", "#76c277", "#3bab5a"]

# 常见异常“国家”噪声（来自地址误切分后的姓名/词）
NOISE_TOKENS = {
    "wei", "jing", "yu", "yang", "xin", "jie", "yan", "lei", "hao", "ying", "li", "yi", "wang",
    "zhang", "liu", "chen", "xu", "sun", "wu", "lin", "zhao", "huang", "zhou", "gao", "ma",
}

# 国家别名统一
ALIAS_MAP = {
    "u.s.a": "usa",
    "u s a": "usa",
    "us": "usa",
    "united states": "usa",
    "united states of america": "usa",
    "america": "usa",
    "england": "uk",
    "scotland": "uk",
    "wales": "uk",
    "northern ireland": "uk",
    "peoples r china": "china",
    "people s r china": "china",
    "peoples republic of china": "china",
}

# 允许的短国家名（避免把 2~3 字母都误删）
SHORT_VALID = {"usa", "uk", "uae", "laos", "oman", "peru", "mali", "chad", "togo", "nepal", "niger", "benin", "qatar"}


def setup_fonts(simsun: str, times: str):
    if os.path.exists(simsun):
        fm.fontManager.addfont(simsun)
    if os.path.exists(times):
        fm.fontManager.addfont(times)
    plt.rcParams["font.sans-serif"] = ["SimSun", "Times New Roman", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def normalize_partner(x: str):
    if pd.isna(x):
        return ""
    t = re.sub(r"\s+", " ", str(x).strip().lower().strip(". ;,"))
    t = ALIAS_MAP.get(t, t)
    return t


def is_valid_partner(name: str):
    if not name:
        return False
    if name in NOISE_TOKENS:
        return False
    # 过滤纯数字或奇怪符号
    if re.fullmatch(r"[\d\W_]+", name):
        return False
    # 太短且不在白名单
    if len(name) <= 3 and name not in SHORT_VALID:
        return False
    return True


def clean_partner_df(df: pd.DataFrame, col="partner"):
    d = df.copy()
    d[col] = d[col].apply(normalize_partner)
    d = d[d[col].apply(is_valid_partner)]
    return d


def plot_top20(metrics_dir: str, out_dir: str):
    src = os.path.join(metrics_dir, "metric_partner_year.csv")
    if os.path.exists(src):
        d = pd.read_csv(src)
        d = clean_partner_df(d, col="partner")
        top = d.groupby("partner", as_index=False)["papers"].sum().sort_values("papers", ascending=False).head(20)
    else:
        # 兜底：直接读 top20 并清洗
        d = pd.read_csv(os.path.join(metrics_dir, "metric_top20_partners.csv"))
        d = clean_partner_df(d, col="partner")
        top = d.groupby("partner", as_index=False)["papers"].sum().sort_values("papers", ascending=False).head(20)

    top = top.sort_values("papers", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(top["partner"], top["papers"], color=BLUE_GRADIENT[2])
    ax.set_title("图6（优化）中国学术合作Top20国家/地区频次")
    ax.set_xlabel("合作论文频次")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig6_top20_partners_refined.png"), dpi=300)
    plt.close(fig)


def plot_heatmap(metrics_dir: str, out_dir: str):
    d = pd.read_csv(os.path.join(metrics_dir, "metric_partner_year.csv"))
    d = clean_partner_df(d, col="partner")

    top10 = d.groupby("partner", as_index=False)["papers"].sum().sort_values("papers", ascending=False).head(10)["partner"].tolist()
    hm = d[d["partner"].isin(top10)].pivot_table(index="partner", columns="PY", values="papers", aggfunc="sum", fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        hm,
        cmap=sns.light_palette(BLUE_GRADIENT[2], as_cmap=True),
        linewidths=0.2,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("图7（优化）中国与主要合作伙伴年度合作强度热力图")
    ax.set_xlabel("年份")
    ax.set_ylabel("合作国家/地区")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig7_partner_heatmap_refined.png"), dpi=300)
    plt.close(fig)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    setup_fonts(args.simsun, args.times)

    plot_top20(args.metrics_dir, args.out_dir)
    plot_heatmap(args.metrics_dir, args.out_dir)

    print("完成：已生成优化版图6和图7")
    print(f"输出目录：{args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于 metrics 结果重绘优化图（图6/图7）")
    parser.add_argument("--metrics_dir", default="./output_funding_analysis/metrics")
    parser.add_argument("--out_dir", default="./output_funding_analysis/figures_refined")
    parser.add_argument("--simsun", default="/data/student2601/WOS/SimSun.ttf")
    parser.add_argument("--times", default="/data/student2601/WOS/Times New Roman.ttf")
    args = parser.parse_args()
    main(args)
