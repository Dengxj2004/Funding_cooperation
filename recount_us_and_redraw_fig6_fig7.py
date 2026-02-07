import argparse
import os
import re
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BLUE_GRADIENT = ["#9bc9dd", "#6baed5", "#4392c4"]

CHINA_ALIASES = {
    "peoples r china", "people s r china", "peoples republic of china", "china",
    "hong kong", "hong kong sar", "hong kong, china",
    "macau", "macao", "macau, china", "macao, china",
    "taiwan", "taiwan, china",
}
USA_ALIASES = {"usa", "u.s.a", "u s a", "us", "united states", "united states of america"}
NOISE_TOKENS = {
    "wei", "jing", "yu", "yang", "xin", "jie", "yan", "lei", "hao", "ying", "li", "yi", "wang",
    "zhang", "liu", "chen", "xu", "sun", "wu", "lin", "zhao", "huang", "zhou", "gao", "ma",
}


def setup_fonts(simsun: str, times: str):
    if os.path.exists(simsun):
        fm.fontManager.addfont(simsun)
    if os.path.exists(times):
        fm.fontManager.addfont(times)
    plt.rcParams["font.sans-serif"] = ["SimSun", "Times New Roman", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def normalize_country_text(x: str):
    t = re.sub(r"\s+", " ", str(x).strip().lower().strip(". ;,"))
    if t in CHINA_ALIASES:
        return "peoples r china"
    if t in USA_ALIASES:
        return "usa"
    # UK统一
    if t in {"england", "scotland", "wales", "northern ireland", "uk", "u.k.", "united kingdom"}:
        return "uk"
    return t


def parse_country_from_address(addr: str):
    """
    修复点：地址末尾可能是 'NJ 08544 USA'，不能直接取最后一个逗号段。
    规则：
    1) 先看整段末尾是否包含常见国家别名（USA/China等）
    2) 再看最后逗号段，去掉邮编后取尾部英文词
    """
    if not addr:
        return "unknown"
    a = str(addr).strip()
    if a.startswith("[") and "]" in a:
        a = a.split("]", 1)[1].strip(" ,")

    low = a.lower().strip(". ;")

    # 快速匹配末尾国家（覆盖USA带州和邮编情况）
    end_patterns = [
        (r"(?:^|\s)(u\.?s\.?a?|united states(?: of america)?)$", "usa"),
        (r"(?:^|\s)(peoples r china|people s r china|peoples republic of china|china)$", "peoples r china"),
        (r"(?:^|\s)(hong kong|hong kong sar|macau|macao|taiwan)(?:,\s*china)?$", "peoples r china"),
        (r"(?:^|\s)(england|scotland|wales|northern ireland|uk|united kingdom)$", "uk"),
    ]
    for pat, val in end_patterns:
        if re.search(pat, low):
            return val

    tail = a.split(",")[-1].strip()
    tail = re.sub(r"\s+", " ", tail)
    # 例如: NJ 08544 USA -> USA
    m = re.search(r"([A-Za-z][A-Za-z .\-]{1,})$", tail)
    if m:
        candidate = normalize_country_text(m.group(1))
    else:
        candidate = normalize_country_text(tail)

    return candidate if candidate else "unknown"


def split_addresses(c1: str):
    if pd.isna(c1) or not str(c1).strip():
        return []
    return [p.strip() for p in str(c1).split(";") if p.strip()]


def parse_row_countries(c1: str):
    addrs = split_addresses(c1)
    return sorted({parse_country_from_address(a) for a in addrs if a})


def read_data(path):
    df = pd.read_csv(path, sep=None, engine="python", dtype=str, low_memory=False)
    for c in ["C1", "PY", "UT", "DI", "TI"]:
        if c not in df.columns:
            df[c] = ""
    return df[["C1", "PY", "UT", "DI", "TI"]].copy()


def build_partner_year(df: pd.DataFrame, n_jobs: int):
    df = df.copy()
    df["PY"] = pd.to_numeric(df["PY"], errors="coerce")
    df = df[df["PY"].notna() & (df["PY"] >= 1995)]
    df["PY"] = df["PY"].astype(int)

    ut = df["UT"].fillna("").str.strip().str.lower()
    di = df["DI"].fillna("").str.strip().str.lower()
    ti = df["TI"].fillna("").str.strip().str.lower()
    fallback = "fallback_" + df.index.astype(str)
    df["paper_id"] = ut.where(ut.ne(""), di.where(di.ne(""), ti.where(ti.ne(""), fallback)))

    c1_list = df["C1"].fillna("").tolist()
    if n_jobs <= 1:
        countries_list = [parse_row_countries(x) for x in c1_list]
    else:
        with Pool(processes=n_jobs) as pool:
            countries_list = list(pool.imap(parse_row_countries, c1_list, chunksize=2500))
    df["countries"] = countries_list

    # 论文级去重并合并国家
    p_rows = []
    for pid, g in df.groupby("paper_id", sort=False):
        py = int(g["PY"].iloc[0])
        cs = sorted(set().union(*g["countries"].tolist()))
        p_rows.append((pid, py, cs))

    partner_year = defaultdict(Counter)
    partner_total = Counter()
    for _, py, cs in p_rows:
        cset = set(cs)
        if "peoples r china" in cset and len(cset) > 1:
            for p in cset - {"peoples r china"}:
                if p in NOISE_TOKENS:
                    continue
                partner_year[py][p] += 1
                partner_total[p] += 1

    py_df = pd.DataFrame([
        {"PY": y, "partner": p, "papers": v}
        for y, c in partner_year.items() for p, v in c.items()
    ])
    top20 = pd.DataFrame(partner_total.items(), columns=["partner", "papers"]).sort_values("papers", ascending=False).head(20)
    return py_df, top20


def plot_fig6(top20: pd.DataFrame, out_dir: str):
    d = top20.sort_values("papers", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(d["partner"], d["papers"], color=BLUE_GRADIENT[2])
    ax.set_title("图6 中国学术合作Top20国家/地区频次（重算）")
    ax.set_xlabel("合作论文频次")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig6_top20_partners_recount.png"), dpi=300)
    plt.close(fig)


def plot_fig7(py_df: pd.DataFrame, out_dir: str):
    top10 = py_df.groupby("partner", as_index=False)["papers"].sum().sort_values("papers", ascending=False).head(10)["partner"].tolist()
    hm = py_df[py_df["partner"].isin(top10)].pivot_table(index="partner", columns="PY", values="papers", aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(hm, cmap=sns.light_palette(BLUE_GRADIENT[2], as_cmap=True), linewidths=0.2, linecolor="white", ax=ax)
    ax.set_title("图7 中国与主要合作伙伴年度合作强度热力图（重算）")
    ax.set_xlabel("年份")
    ax.set_ylabel("国家/地区")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig7_partner_heatmap_recount.png"), dpi=300)
    plt.close(fig)


def main(args):
    setup_fonts(args.simsun, args.times)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.metrics_out_dir, exist_ok=True)

    n_jobs = args.n_jobs if args.n_jobs > 0 else max(1, cpu_count() - 1)

    df = pd.concat([read_data(args.file1), read_data(args.file2)], ignore_index=True)
    py_df, top20 = build_partner_year(df, n_jobs=n_jobs)

    py_df.to_csv(os.path.join(args.metrics_out_dir, "metric_partner_year_recount.csv"), index=False, encoding="utf-8-sig")
    top20.to_csv(os.path.join(args.metrics_out_dir, "metric_top20_partners_recount.csv"), index=False, encoding="utf-8-sig")

    usa_count = int(top20[top20["partner"] == "usa"]["papers"].sum()) if len(top20) else 0
    with open(os.path.join(args.metrics_out_dir, "usa_recount_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"USA collaboration papers (recount): {usa_count}\n")
        f.write(f"n_jobs: {n_jobs}\n")

    plot_fig6(top20, args.out_dir)
    plot_fig7(py_df, args.out_dir)

    print(f"重算完成。USA合作次数（Top20内）: {usa_count}")
    print(f"图输出: {args.out_dir}")
    print(f"指标输出: {args.metrics_out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="重算中美等合作次数并重绘图6/图7（修复邮编+USA地址）")
    parser.add_argument("--file1", default="/data/student2601/WOS/WOS/Funding/merged_paper_info_with_details_full.tsv")
    parser.add_argument("--file2", default="/data/student2601/WOS/WOS/Funding/2022merged_paper_info_new_unique.csv")
    parser.add_argument("--out_dir", default="./output_funding_analysis/figures_recount")
    parser.add_argument("--metrics_out_dir", default="./output_funding_analysis/metrics_recount")
    parser.add_argument("--simsun", default="/data/student2601/WOS/SimSun.ttf")
    parser.add_argument("--times", default="/data/student2601/WOS/Times New Roman.ttf")
    parser.add_argument("--n_jobs", type=int, default=0, help="并行进程数，0=CPU核数-1")
    args = parser.parse_args()
    main(args)
