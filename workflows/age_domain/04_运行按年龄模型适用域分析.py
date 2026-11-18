#!/usr/bin/env python3
"""最终冻结锆石数据的按年龄模型适用域分析。

本程序只读取冻结预测，不训练、不调参、不重新预测，也不会修改输入文件。
直接在PyCharm中运行本文件即可。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DEFAULT = ROOT / "inputs/01_论文最终逐颗锆石数据_10879条_FROZEN.csv"
OUTPUT_DEFAULT = ROOT / "按年龄模型适用域结果"

AGE = "AGE_MA_CLEAN"
SOURCE = "SOURCE_ID"
REFERENCE = "REFERENCE_ID"
PRED = "DELTA_FMQ_ENSEMBLE"
MLP = "DELTA_FMQ_MLP"
ET = "DELTA_FMQ_EXTRATREES"
MODEL_DIFF = "MODEL_ABS_DIFF"
DOMAIN_DISTANCE = "DOMAIN_NN_DISTANCE"
DOMAIN_THRESHOLD = "DOMAIN_THRESHOLD"
DOMAIN_PASS = "DOMAIN_PASS"
FEATURE_COUNT = "N_FEATURES_MEASURED"

STAGES = [
    ("0–400 Ma", 0, 400),
    ("400–800 Ma", 400, 800),
    ("800–1800 Ma", 800, 1800),
    ("1800–2500 Ma", 1800, 2500),
    ("2500–3000 Ma", 2500, 3000),
    ("3000–3500 Ma", 3000, 3500),
    ("3500–4000 Ma", 3500, 4001),
]

CONTRASTS = [
    ("古老地球氧化上升", "3000–3300减3500–4000", ">0"),
    ("晚太古代高值", "2500–3000减1800–2500", ">0"),
    ("中元古代低值", "800–1800减两侧均值", "<0"),
    ("显生宙恢复", "0–400减800–1800", ">0"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="按年龄审计模型适用域、模型一致性和元素完整性；不会训练模型"
    )
    parser.add_argument("--data", default=str(DATA_DEFAULT))
    parser.add_argument("--output", default=str(OUTPUT_DEFAULT))
    return parser.parse_args()


def setup_plot():
    plt.rcParams.update({
        "font.sans-serif": [
            "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"
        ],
        "axes.unicode_minus": False,
        "figure.dpi": 130,
        "savefig.dpi": 220,
    })


def savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    q = df.copy()
    for col in q.columns:
        q[col] = q[col].map(
            lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x)
        )
    head = "| " + " | ".join(map(str, q.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(q.columns)) + "|"
    rows = ["| " + " | ".join(row) + " |" for row in q.astype(str).to_numpy()]
    return "\n".join([head, sep] + rows)


def load_and_validate(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    required = [
        "GLOBAL_RECORD_ID", AGE, SOURCE, REFERENCE, PRED, MLP, ET, MODEL_DIFF,
        DOMAIN_DISTANCE, DOMAIN_THRESHOLD, DOMAIN_PASS, FEATURE_COUNT,
    ]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise RuntimeError(f"冻结数据缺少字段：{missing}")
    if len(d) != 10879 or d["GLOBAL_RECORD_ID"].nunique() != 10879:
        raise RuntimeError("输入文件不是正式冻结的10879条唯一记录")
    if d[required].isna().any().any():
        bad = d[required].isna().sum()
        raise RuntimeError(f"关键字段存在空值：{bad[bad > 0].to_dict()}")
    if not d[DOMAIN_PASS].astype(bool).all():
        raise RuntimeError("冻结数据中出现未通过适用域筛选的记录，请核对文件版本")

    d = d.copy()
    d["DOMAIN_DISTANCE_RATIO"] = d[DOMAIN_DISTANCE] / d[DOMAIN_THRESHOLD]
    d["NEAR_DOMAIN_BOUNDARY_0_8"] = d["DOMAIN_DISTANCE_RATIO"] >= .80
    d["VERY_NEAR_DOMAIN_BOUNDARY_0_9"] = d["DOMAIN_DISTANCE_RATIO"] >= .90
    d["COMPLETE_14"] = d[FEATURE_COUNT].eq(14)
    d["MISSING_1_TO_2"] = d[FEATURE_COUNT].between(12, 13)
    return d


def source_equal_median(g: pd.DataFrame, value_col=PRED):
    if g.empty:
        return np.nan
    return float(g.groupby(SOURCE)[value_col].median().median())


def support_label(n_particles: int, n_sources: int) -> str:
    """仅用于醒目标记稀疏窗口，不作为统计显著性或删除标准。"""
    if n_particles < 50 or n_sources < 5:
        return "稀疏"
    if n_particles < 100 or n_sources < 10:
        return "中等"
    return "较充足"


def summarize_group(g: pd.DataFrame, label, start, end):
    counts = g[SOURCE].value_counts()
    return {
        "年龄区间": label,
        "年龄起点_Ma": float(start),
        "年龄终点_Ma": float(end),
        "年龄中心_Ma": float((start + min(end, 4000)) / 2),
        "颗粒数": int(len(g)),
        "独立来源数": int(g[SOURCE].nunique()),
        "独立文献数": int(g[REFERENCE].nunique()),
        "最大来源占比": float(counts.iloc[0] / len(g)) if len(g) else np.nan,
        "数据支撑标记": support_label(len(g), g[SOURCE].nunique()),
        "原适用域通过率": float(g[DOMAIN_PASS].mean()),
        "归一化域距离_中位数": float(g["DOMAIN_DISTANCE_RATIO"].median()),
        "归一化域距离_P90": float(g["DOMAIN_DISTANCE_RATIO"].quantile(.90)),
        "归一化域距离_P95": float(g["DOMAIN_DISTANCE_RATIO"].quantile(.95)),
        "接近边界比例_ge0.8": float(g["NEAR_DOMAIN_BOUNDARY_0_8"].mean()),
        "非常接近边界比例_ge0.9": float(g["VERY_NEAR_DOMAIN_BOUNDARY_0_9"].mean()),
        "14元素完整率": float(g["COMPLETE_14"].mean()),
        "缺1至2元素比例": float(g["MISSING_1_TO_2"].mean()),
        "模型差异_中位数": float(g[MODEL_DIFF].median()),
        "模型差异_P90": float(g[MODEL_DIFF].quantile(.90)),
        "模型差异_P95": float(g[MODEL_DIFF].quantile(.95)),
        "模型差异_ge0.5比例": float((g[MODEL_DIFF] >= .5).mean()),
        "模型差异_ge1.0比例": float((g[MODEL_DIFF] >= 1.0).mean()),
        "MLP颗粒中位数": float(g[MLP].median()),
        "ExtraTrees颗粒中位数": float(g[ET].median()),
        "双模型颗粒中位数": float(g[PRED].median()),
        "来源等权双模型中位数": source_equal_median(g),
    }


def summarize_100ma(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for start in range(0, 4000, 100):
        end = start + 100
        g = d[(d[AGE] >= start) & (d[AGE] < end)]
        if len(g):
            rows.append(summarize_group(g, f"{start}–{end}", start, end))
    return pd.DataFrame(rows)


def summarize_stages(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, start, end in STAGES:
        g = d[(d[AGE] >= start) & (d[AGE] < end)]
        rows.append(summarize_group(g, label, start, end))
    return pd.DataFrame(rows)


def source_audit(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, g in d.groupby(SOURCE, sort=True):
        rows.append({
            "SOURCE_ID": source,
            "REFERENCE_ID": "; ".join(sorted(g[REFERENCE].astype(str).unique())),
            "颗粒数": int(len(g)),
            "年龄最小_Ma": float(g[AGE].min()),
            "年龄最大_Ma": float(g[AGE].max()),
            "归一化域距离_中位数": float(g["DOMAIN_DISTANCE_RATIO"].median()),
            "归一化域距离_P90": float(g["DOMAIN_DISTANCE_RATIO"].quantile(.90)),
            "接近边界比例_ge0.8": float(g["NEAR_DOMAIN_BOUNDARY_0_8"].mean()),
            "14元素完整率": float(g["COMPLETE_14"].mean()),
            "模型差异_中位数": float(g[MODEL_DIFF].median()),
            "模型差异_P90": float(g[MODEL_DIFF].quantile(.90)),
        })
    return pd.DataFrame(rows).sort_values(
        ["归一化域距离_P90", "颗粒数"], ascending=[False, False]
    )


def interval_source_equal(d: pd.DataFrame, lo: float, hi: float) -> float:
    g = d[(d[AGE] >= lo) & (d[AGE] < hi)]
    return source_equal_median(g)


def key_contrasts(d: pd.DataFrame):
    values = {
        "young": interval_source_equal(d, 0, 400),
        "pre_mid": interval_source_equal(d, 400, 800),
        "mid": interval_source_equal(d, 800, 1800),
        "post_mid": interval_source_equal(d, 1800, 2500),
        "late_arch": interval_source_equal(d, 2500, 3000),
        "ancient_young": interval_source_equal(d, 3000, 3300),
        "ancient_old": interval_source_equal(d, 3500, 4001),
    }
    return {
        "古老地球氧化上升": values["ancient_young"] - values["ancient_old"],
        "晚太古代高值": values["late_arch"] - values["post_mid"],
        "中元古代低值": values["mid"] - (values["pre_mid"] + values["post_mid"]) / 2,
        "显生宙恢复": values["young"] - values["mid"],
    }


def subset_comparison(d: pd.DataFrame):
    subsets = [
        ("全部冻结数据", d),
        ("域内更深_距离比小于0.8", d[d["DOMAIN_DISTANCE_RATIO"] < .80]),
        ("完整14元素", d[d["COMPLETE_14"]]),
        (
            "域内更深且完整14元素",
            d[(d["DOMAIN_DISTANCE_RATIO"] < .80) & d["COMPLETE_14"]],
        ),
    ]
    contrast_rows = []
    curve_rows = []
    for name, q in subsets:
        contrast = key_contrasts(q)
        for conclusion, definition, direction in CONTRASTS:
            contrast_rows.append({
                "数据情景": name,
                "颗粒数": int(len(q)),
                "来源数": int(q[SOURCE].nunique()),
                "结论": conclusion,
                "对比定义": definition,
                "预期方向": direction,
                "来源等权差值": contrast[conclusion],
                "方向是否保留": bool(
                    contrast[conclusion] > 0 if direction == ">0" else contrast[conclusion] < 0
                ),
            })
        for start in range(0, 4000, 100):
            g = q[(q[AGE] >= start) & (q[AGE] < start + 100)]
            curve_rows.append({
                "数据情景": name,
                "年龄窗起点_Ma": start,
                "年龄中心_Ma": start + 50,
                "颗粒数": int(len(g)),
                "来源数": int(g[SOURCE].nunique()),
                "来源等权双模型中位数": source_equal_median(g),
            })
    return pd.DataFrame(contrast_rows), pd.DataFrame(curve_rows)


def make_figures(t100, stages, comparison_curve, d, out):
    setup_plot()
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    x = t100["年龄中心_Ma"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(x, t100["归一化域距离_中位数"], marker="o", ms=3, label="中位数")
    ax.plot(x, t100["归一化域距离_P90"], color="#e45756", label="P90")
    ax.axhline(.8, color="#dd6b20", linestyle="--", label="边界距离的80%")
    ax.axhline(1.0, color="black", linestyle=":", label="原适用域边界")
    ax.set(xlabel="年龄（Ma，向右更古老）", ylabel="最近邻距离/适用域阈值",
           title="每100 Ma窗口的特征空间适用域位置")
    ax.legend(); ax.grid(alpha=.2)
    savefig(fig, fig_dir / "01_按年龄归一化适用域距离.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    axes[0].plot(x, t100["模型差异_中位数"], label="中位数")
    axes[0].plot(x, t100["模型差异_P90"], color="#e45756", label="P90")
    axes[0].axhline(.5, color="#dd6b20", linestyle="--", linewidth=1)
    axes[0].set_ylabel("|MLP－ExtraTrees| ΔFMQ")
    axes[0].set_title("按年龄的双模型差异")
    axes[0].legend()
    axes[1].plot(x, t100["14元素完整率"], color="#2f855a", label="14元素完整率")
    axes[1].plot(x, t100["接近边界比例_ge0.8"], color="#805ad5", label="接近适用域边界比例")
    axes[1].set(xlabel="年龄（Ma）", ylabel="比例", ylim=(0, 1.03))
    axes[1].legend()
    savefig(fig, fig_dir / "02_按年龄模型差异与数据完整性.png")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(x, t100["颗粒数"]); axes[0].set_ylabel("颗粒数")
    axes[0].set_yscale("log")
    axes[1].plot(x, t100["独立来源数"], color="#2f855a"); axes[1].set_ylabel("来源数")
    axes[2].plot(x, t100["最大来源占比"], color="#dd6b20"); axes[2].set_ylabel("最大来源占比")
    axes[2].set_xlabel("年龄（Ma）")
    fig.suptitle("每100 Ma窗口的数据支撑度")
    savefig(fig, fig_dir / "03_按年龄颗粒来源与集中度.png")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for name, g in comparison_curve.groupby("数据情景", sort=False):
        ax.plot(
            g["年龄中心_Ma"], g["来源等权双模型中位数"],
            marker="o", ms=2.5, linewidth=1.3, label=name,
        )
    ax.set(xlabel="年龄（Ma）", ylabel="来源等权预测 ΔFMQ",
           title="全部数据与适用域核心子集的100 Ma趋势")
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    savefig(fig, fig_dir / "04_适用域核心子集趋势对比.png")

    old = t100[t100["年龄起点_Ma"] >= 3000]
    xo = old["年龄中心_Ma"].to_numpy(float)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    axes[0].bar(xo, old["独立来源数"], width=70, color="#4c78a8")
    axes[0].set_ylabel("来源数")
    axes[1].plot(xo, old["归一化域距离_中位数"], marker="o", label="中位数")
    axes[1].plot(xo, old["归一化域距离_P90"], marker="o", label="P90")
    axes[1].axhline(1.0, color="black", linestyle=":")
    axes[1].set_ylabel("归一化域距离"); axes[1].legend()
    axes[2].plot(xo, old["模型差异_中位数"], marker="o", label="中位数")
    axes[2].plot(xo, old["模型差异_P90"], marker="o", label="P90")
    axes[2].set(xlabel="年龄（Ma）", ylabel="模型差异 ΔFMQ"); axes[2].legend()
    fig.suptitle("3000–4000 Ma太古宙重点适用性审计")
    savefig(fig, fig_dir / "05_太古宙重点适用性审计.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(t100["年龄中心_Ma"], t100["MLP颗粒中位数"], label="MLP")
    axes[0].plot(t100["年龄中心_Ma"], t100["ExtraTrees颗粒中位数"], label="ExtraTrees")
    axes[0].set(title="两冻结模型的年龄趋势", xlabel="年龄（Ma）", ylabel="预测 ΔFMQ")
    axes[0].legend()
    axes[1].scatter(d["DOMAIN_DISTANCE_RATIO"], d[MODEL_DIFF], s=5, alpha=.15)
    axes[1].set(xlabel="归一化域距离", ylabel="两模型绝对差 ΔFMQ",
                title="接近适用域边界是否伴随模型分歧")
    savefig(fig, fig_dir / "06_模型趋势与适用域距离关系.png")


def write_report(d, t100, stages, contrasts, sources, out):
    old = t100[t100["年龄起点_Ma"] >= 3500]
    sparse = t100[t100["数据支撑标记"].eq("稀疏")][[
        "年龄区间", "颗粒数", "独立来源数", "最大来源占比",
        "归一化域距离_中位数", "模型差异_中位数",
    ]]
    core = contrasts[[
        "数据情景", "结论", "预期方向", "来源等权差值", "方向是否保留"
    ]]
    top_boundary = sources.head(8)[[
        "SOURCE_ID", "颗粒数", "年龄最小_Ma", "年龄最大_Ma",
        "归一化域距离_P90", "模型差异_P90",
    ]]
    report = f"""# 按年龄模型适用域与外推风险分析报告

## 运行信息

- 冻结数据：{len(d):,}颗锆石，{d[SOURCE].nunique()}个来源，{d[REFERENCE].nunique()}篇文献
- 年龄范围：{d[AGE].min():.3f}–{d[AGE].max():.1f} Ma
- 原适用域阈值通过率：{d[DOMAIN_PASS].mean():.1%}
- 分析方式：100 Ma窗口、宽年龄阶段、3000–4000 Ma太古宙重点审计
- 本程序未训练、未调参、未重新预测

## 如何理解100%适用域通过率

最终数据在冻结前已经执行适用域筛选，因此10,879条记录全部通过原阈值。
这不能解释成所有年龄段均不存在外推风险。本报告进一步使用：

1. 最近训练样本距离与阈值之比；
2. MLP与ExtraTrees预测差异；
3. 元素完整率；
4. 每窗颗粒数、来源数和最大来源占比；
5. 域内更深核心子集趋势；

共同评估按年龄变化的适用性。

## 宽年龄阶段汇总

{markdown_table(stages[[
    '年龄区间', '颗粒数', '独立来源数', '最大来源占比',
    '归一化域距离_中位数', '归一化域距离_P90', '接近边界比例_ge0.8',
    '14元素完整率', '模型差异_中位数', '模型差异_P90'
]])}

## 全部数据与适用域核心子集的关键趋势

{markdown_table(core)}

## 稀疏100 Ma窗口

{markdown_table(sparse)}

## 最接近适用域边界的8个来源

{markdown_table(top_boundary)}

## 太古宙解释原则

- 3000–4000 Ma结果应同时结合来源数、域距离和双模型差异解释。
- 即使元素特征仍位于训练特征域内，也不能替代同年龄真值验证。
- 若全部数据与“域内更深且完整14元素”子集保持相同方向，可说明趋势不由边界样本或缺失值主导。
- >3500 Ma的局部100 Ma峰谷不作为强结论，重点解释更宽时间尺度的总体方向。

## 3500–4000 Ma窗口概览

{markdown_table(old[[
    '年龄区间', '颗粒数', '独立来源数', '最大来源占比',
    '归一化域距离_中位数', '归一化域距离_P90',
    '模型差异_中位数', '模型差异_P90'
]])}
"""
    (out / "07_按年龄模型适用域分析报告.md").write_text(report, encoding="utf-8")


def main():
    args = parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    d = load_and_validate(Path(args.data).resolve())
    print("数据核验通过：10,879颗锆石；冻结数据不会被修改。")

    t100 = summarize_100ma(d)
    stages = summarize_stages(d)
    sources = source_audit(d)
    contrasts, curves = subset_comparison(d)
    archean = t100[t100["年龄起点_Ma"] >= 3000].copy()

    t100.to_csv(out / "01_每100Ma适用域与数据支撑.csv", index=False, encoding="utf-8-sig")
    stages.to_csv(out / "02_宽年龄阶段适用域汇总.csv", index=False, encoding="utf-8-sig")
    archean.to_csv(out / "03_太古宙3000至4000Ma重点审计.csv", index=False, encoding="utf-8-sig")
    contrasts.to_csv(out / "04_核心子集关键趋势对比.csv", index=False, encoding="utf-8-sig")
    curves.to_csv(out / "05_核心子集100Ma曲线.csv", index=False, encoding="utf-8-sig")
    sources.to_csv(out / "06_逐来源适用域审计.csv", index=False, encoding="utf-8-sig")

    make_figures(t100, stages, curves, d, out)
    write_report(d, t100, stages, contrasts, sources, out)
    manifest = {
        "status": "COMPLETED",
        "training_performed": False,
        "prediction_performed": False,
        "input_rows": int(len(d)),
        "sources": int(d[SOURCE].nunique()),
        "references": int(d[REFERENCE].nunique()),
        "original_domain_pass_rate": float(d[DOMAIN_PASS].mean()),
        "central_domain_definition": "DOMAIN_NN_DISTANCE / DOMAIN_THRESHOLD < 0.8",
        "window_width_ma": 100,
    }
    (out / "08_运行清单.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"完成。结果保存在：{out}")
    print("请把整个‘按年龄模型适用域结果’文件夹压缩后发回分析。")


if __name__ == "__main__":
    main()
