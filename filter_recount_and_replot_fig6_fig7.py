import argparse
import os
import re

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BLUE_GRADIENT = ["#9bc9dd", "#6baed5", "#4392c4"]

# 常见姓名噪声（可继续扩展）
NAME_NOISE = {
    "wei", "jing", "yu", "yang", "xin", "jie", "yan", "lei", "hao", "ying", "li", "yi",
    "jun", "hui", "peng", "tao", "jian", "rui", "bo", "yue", "chao",
    "wang", "zhang", "liu", "chen", "xu", "sun", "wu", "lin", "zhao", "huang", "zhou", "gao", "ma",
}

# 国家别名规范化
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

SHORT_VALID = {"usa", "uk", "uae", "oman", "peru", "mali", "chad", "togo", "laos", "qatar", "niger", "benin"}


def setup_fonts(simsun: str, times: str):
    if os.path.exists(simsun):
        fm.fontManager.addfont(simsun)
    if os.path.exists(times):
        fm.fontManager.addfont(times)
    plt.rcParams["font.sans-serif"] = ["SimSun", "Times New Roman", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def norm_partner(x: str):
    t = re.sub(r"\s+", " ", str(x).strip().lower().strip(". ;,"))
    return ALIAS_MAP.get(t, t)


def is_valid_partner(x: str):
    if not x:
        return False
    if x in NAME_NOISE:
        return False
    if re.fullmatch(r"[\d\W_]+", x):
        return False
    # 长度极短且不在白名单
    if len(x) <= 3 and x not in SHORT_VALID:
        return False
    # 两段拼音姓名模式（如 yu qing）
    if re.fullmatch(r"[a-z]{2,8}( [a-z]{2,8}){1,2}", x) and all(len(i) <= 8 for i in x.split()):
        if x.replace(" ", "") not in {"southkorea", "newzealand", "saudiarabia", "unitedkingdom"}:
            return False
    return True


def clean_partner_year(df: pd.DataFrame):
    d = df.copy()
    d["partner"] = d["partner"].apply(norm_partner)
    d = d[d["partner"].apply(is_valid_partner)]
    d = d.groupby(["PY", "partner"], as_index=False)["papers"].sum()
    return d


def plot_fig6(top20: pd.DataFrame, out_dir: str):
    d = top20.sort_values("papers", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(d["partner"], d["papers"], color=BLUE_GRADIENT[2])
    ax.set_title("图6 中国学术合作Top20国家/地区频次（去噪后）")
    ax.set_xlabel("合作论文频次")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig6_top20_partners_recount_filtered.png"), dpi=300)
    plt.close(fig)


def plot_fig7(py_df: pd.DataFrame, out_dir: str):
    top10 = py_df.groupby("partner", as_index=False)["papers"].sum().sort_values("papers", ascending=False).head(10)["partner"].tolist()
    hm = py_df[py_df["partner"].isin(top10)].pivot_table(index="partner", columns="PY", values="papers", aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(hm, cmap=sns.light_palette(BLUE_GRADIENT[2], as_cmap=True), linewidths=0.2, linecolor="white", ax=ax)
    ax.set_title("图7 中国与主要合作伙伴年度合作强度热力图（去噪后）")
    ax.set_xlabel("年份")
    ax.set_ylabel("国家/地区")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig7_partner_heatmap_recount_filtered.png"), dpi=300)
    plt.close(fig)


def main(args):
    setup_fonts(args.simsun, args.times)
    os.makedirs(args.out_dir, exist_ok=True)

    src = os.path.join(args.metrics_recount_dir, "metric_partner_year_recount.csv")
    if not os.path.exists(src):
        raise FileNotFoundError(f"未找到文件: {src}")

    py = pd.read_csv(src)
    clean = clean_partner_year(py)

    top20 = clean.groupby("partner", as_index=False)["papers"].sum().sort_values("papers", ascending=False).head(20)
    clean.to_csv(os.path.join(args.out_dir, "metric_partner_year_recount_filtered.csv"), index=False, encoding="utf-8-sig")
    top20.to_csv(os.path.join(args.out_dir, "metric_top20_partners_recount_filtered.csv"), index=False, encoding="utf-8-sig")

    removed = sorted(set(py["partner"].astype(str).str.lower()) - set(clean["partner"].astype(str).str.lower()))
    with open(os.path.join(args.out_dir, "removed_partner_tokens.txt"), "w", encoding="utf-8") as f:
        for x in removed:
            f.write(f"{x}\n")

    plot_fig6(top20, args.out_dir)
    plot_fig7(clean, args.out_dir)

    print("完成：已从重算结果中排除姓名噪声并重绘图6/图7")
    print(f"输出目录：{args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从已有 metric_partner_year_recount.csv 去噪并重绘图6/图7")
    parser.add_argument("--metrics_recount_dir", default="./output_funding_analysis/metrics_recount")
    parser.add_argument("--out_dir", default="./output_funding_analysis/figures_recount_filtered")
    parser.add_argument("--simsun", default="/data/student2601/WOS/SimSun.ttf")
    parser.add_argument("--times", default="/data/student2601/WOS/Times New Roman.ttf")
    args = parser.parse_args()
    main(args)
