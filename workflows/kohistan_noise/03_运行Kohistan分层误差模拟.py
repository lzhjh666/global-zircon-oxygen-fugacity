#!/usr/bin/env python3
"""Kohistan分层误差敏感性检验。

只读取已冻结的全球逐颗锆石预测与Kohistan真实残差；不训练、不调参、
不改写任何冻结数据。直接点击运行或执行 ``python 03_运行Kohistan分层误差模拟.py``。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
GLOBAL_DEFAULT = ROOT / "inputs/01_论文最终逐颗锆石数据_10879条_FROZEN.csv"
KOHISTAN_DEFAULT = ROOT / "inputs/02_Kohistan_113颗_旧冻结双模型真实残差.csv"
OUTPUT_DEFAULT = ROOT / "Kohistan分层误差结果"

AGE_COL = "AGE_MA_CLEAN"
PRED_COL = "DELTA_FMQ_ENSEMBLE"
SOURCE_COL = "SOURCE_ID"
BODY_COL = "岩体"
RESIDUAL_COL = "ENSEMBLE_RESIDUAL_PRED_MINUS_TRUE"

CONTRASTS = [
    ("古老地球氧化上升", ">0"),
    ("晚太古代高值", ">0"),
    ("中元古代低值_相对两侧", "<0"),
    ("显生宙恢复", ">0"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="运行Kohistan颗粒级与来源级分层误差敏感性检验；不会训练模型"
    )
    parser.add_argument("--global-data", default=str(GLOBAL_DEFAULT))
    parser.add_argument("--kohistan-residuals", default=str(KOHISTAN_DEFAULT))
    parser.add_argument("--output", default=str(OUTPUT_DEFAULT))
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261118)
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


def load_inputs(global_path: Path, residual_path: Path):
    global_df = pd.read_csv(global_path, low_memory=False)
    residual_df = pd.read_csv(residual_path, low_memory=False)

    required_global = ["GLOBAL_RECORD_ID", SOURCE_COL, AGE_COL, PRED_COL]
    required_residual = [BODY_COL, RESIDUAL_COL]
    missing_global = [c for c in required_global if c not in global_df.columns]
    missing_residual = [c for c in required_residual if c not in residual_df.columns]
    if missing_global:
        raise RuntimeError(f"全球冻结数据缺少字段：{missing_global}")
    if missing_residual:
        raise RuntimeError(f"Kohistan残差数据缺少字段：{missing_residual}")

    if len(global_df) != 10879 or global_df["GLOBAL_RECORD_ID"].nunique() != 10879:
        raise RuntimeError("全球冻结数据不是10879条唯一记录，请勿使用其他版本替换")
    if global_df[required_global].isna().any().any():
        raise RuntimeError("全球冻结数据关键字段存在空值")

    residual_df[RESIDUAL_COL] = pd.to_numeric(residual_df[RESIDUAL_COL], errors="coerce")
    residual_df = residual_df.dropna(subset=[BODY_COL, RESIDUAL_COL]).copy()
    if len(residual_df) != 113 or residual_df[BODY_COL].nunique() != 5:
        raise RuntimeError("Kohistan残差数据应为113颗、5个岩体")
    return global_df, residual_df


def prepare_groups(global_df: pd.DataFrame):
    source_names = np.array(sorted(global_df[SOURCE_COL].astype(str).unique()))
    source_lookup = {name: i for i, name in enumerate(source_names)}
    source_code = global_df[SOURCE_COL].astype(str).map(source_lookup).to_numpy(int)
    ages = pd.to_numeric(global_df[AGE_COL], errors="raise").to_numpy(float)
    pred = pd.to_numeric(global_df[PRED_COL], errors="raise").to_numpy(float)

    source_rows = [np.flatnonzero(source_code == i) for i in range(len(source_names))]
    starts = np.arange(0, 4000, 100)
    bin_code = np.floor(ages / 100).astype(int)
    bin_rows = [np.flatnonzero((ages >= b) & (ages < b + 100)) for b in starts]

    source_bin_groups = []
    for source_i in range(len(source_names)):
        source_mask = source_code == source_i
        for bin_i, start in enumerate(starts):
            rows = np.flatnonzero(source_mask & (ages >= start) & (ages < start + 100))
            if len(rows):
                source_bin_groups.append((source_i, bin_i, rows))

    interval_bounds = {
        "young": (0, 400),
        "pre_mid": (400, 800),
        "mid": (800, 1800),
        "post_mid": (1800, 2500),
        "late_arch": (2500, 3000),
        "ancient_young": (3000, 3300),
        "ancient_old": (3500, 4001),
    }
    interval_rows = {}
    interval_source_groups = {}
    for name, (lo, hi) in interval_bounds.items():
        interval_rows[name] = np.flatnonzero((ages >= lo) & (ages < hi))
        groups = []
        for source_i in range(len(source_names)):
            rows = np.flatnonzero((source_code == source_i) & (ages >= lo) & (ages < hi))
            if len(rows):
                groups.append(rows)
        interval_source_groups[name] = groups

    return {
        "source_names": source_names,
        "source_code": source_code,
        "source_rows": source_rows,
        "ages": ages,
        "pred": pred,
        "starts": starts,
        "bin_rows": bin_rows,
        "source_bin_groups": source_bin_groups,
        "interval_rows": interval_rows,
        "interval_source_groups": interval_source_groups,
    }


def kohistan_components(residual_df: pd.DataFrame):
    body_names = np.array(sorted(residual_df[BODY_COL].astype(str).unique()))
    body_means = []
    within_pools = []
    rows = []
    for body in body_names:
        residual = residual_df.loc[
            residual_df[BODY_COL].astype(str).eq(body), RESIDUAL_COL
        ].to_numpy(float)
        body_mean = float(np.mean(residual))
        within = residual - body_mean
        body_means.append(body_mean)
        within_pools.append(within)
        rows.append({
            "岩体": body,
            "颗粒数": len(residual),
            "平均残差_预测减真值": body_mean,
            "岩体内残差标准差": float(np.std(within, ddof=1)),
            "MAE": float(np.mean(np.abs(residual))),
            "RMSE": float(np.sqrt(np.mean(residual ** 2))),
        })

    body_means = np.asarray(body_means, float)
    centered_body_means = body_means - np.mean(body_means)
    all_residuals = residual_df[RESIDUAL_COL].to_numpy(float)
    centered_all_residuals = all_residuals - np.mean(all_residuals)
    between_variance = float(np.var(body_means, ddof=1))
    mean_within_variance = float(np.mean([np.var(x, ddof=1) for x in within_pools]))
    icc = between_variance / (between_variance + mean_within_variance)
    decomposition = {
        "Kohistan颗粒数": int(len(residual_df)),
        "Kohistan岩体数": int(len(body_names)),
        "总体bias": float(np.mean(all_residuals)),
        "总体MSE": float(np.mean(all_residuals ** 2)),
        "总体RMSE": float(np.sqrt(np.mean(all_residuals ** 2))),
        "岩体均值间方差": between_variance,
        "平均岩体内方差": mean_within_variance,
        "岩体内相关近似_ICC": float(icc),
        "说明": "ICC仅用于描述Kohistan五岩体的误差聚集程度，不代表全球来源的真实ICC",
    }
    return (
        body_names, centered_body_means, within_pools,
        centered_all_residuals, pd.DataFrame(rows), decomposition,
    )


def particle_curve(values, groups):
    return np.asarray([np.median(values[rows]) for rows in groups], float)


def source_equal_curve(values, prepared):
    matrix = np.full((len(prepared["source_names"]), len(prepared["starts"])), np.nan)
    for source_i, bin_i, rows in prepared["source_bin_groups"]:
        matrix[source_i, bin_i] = np.median(values[rows])
    return np.nanmedian(matrix, axis=0)


def interval_values(values, prepared, weighting):
    if weighting == "颗粒中位数":
        return {
            name: float(np.median(values[rows]))
            for name, rows in prepared["interval_rows"].items()
        }
    result = {}
    for name, groups in prepared["interval_source_groups"].items():
        result[name] = float(np.median([np.median(values[rows]) for rows in groups]))
    return result


def calculate_contrasts(values, prepared, weighting):
    w = interval_values(values, prepared, weighting)
    shoulders = (w["pre_mid"] + w["post_mid"]) / 2
    return {
        "古老地球氧化上升": w["ancient_young"] - w["ancient_old"],
        "晚太古代高值": w["late_arch"] - w["post_mid"],
        "中元古代低值_相对两侧": w["mid"] - shoulders,
        "显生宙恢复": w["young"] - w["mid"],
    }


def independent_error(rng, centered_residuals, n_rows):
    return rng.choice(centered_residuals, size=n_rows, replace=True)


def hierarchical_error(rng, prepared, centered_body_means, within_pools):
    """每个全球来源抽取一个Kohistan岩体偏差，颗粒再抽对应岩体内部残差。"""
    error = np.empty(len(prepared["pred"]), dtype=float)
    selected_body = rng.integers(0, len(centered_body_means), size=len(prepared["source_names"]))
    for source_i, rows in enumerate(prepared["source_rows"]):
        body_i = selected_body[source_i]
        within = rng.choice(within_pools[body_i], size=len(rows), replace=True)
        error[rows] = centered_body_means[body_i] + within
    return error


def robustness_label(probability):
    if probability >= 0.95:
        return "强稳健"
    if probability >= 0.90:
        return "较强支持"
    if probability >= 0.80:
        return "方向性支持"
    return "对系统误差敏感"


def direction_summary(contrast_df: pd.DataFrame, baseline_contrasts):
    rows = []
    for scenario, weighting in contrast_df[["误差情景", "聚合方法"]].drop_duplicates().itertuples(index=False):
        part = contrast_df[
            contrast_df["误差情景"].eq(scenario) & contrast_df["聚合方法"].eq(weighting)
        ]
        for contrast, direction in CONTRASTS:
            probability = float((part[contrast] > 0).mean()) if direction == ">0" else float((part[contrast] < 0).mean())
            baseline = baseline_contrasts[(weighting, contrast)]
            rows.append({
                "误差情景": scenario,
                "聚合方法": weighting,
                "结论": contrast,
                "预期方向": direction,
                "无噪声基线差值": baseline,
                "模拟P2.5": float(part[contrast].quantile(.025)),
                "模拟中位数": float(part[contrast].median()),
                "模拟P97.5": float(part[contrast].quantile(.975)),
                "方向保留概率": probability,
                "稳健性判断": robustness_label(probability),
            })
    return pd.DataFrame(rows)


def run_simulation(global_df, residual_df, iterations, seed, out):
    prepared = prepare_groups(global_df)
    (
        body_names, body_offsets, within_pools, centered_residuals,
        body_stats, decomposition,
    ) = kohistan_components(residual_df)
    body_stats.to_csv(out / "01_Kohistan分岩体残差统计.csv", index=False, encoding="utf-8-sig")
    (out / "02_Kohistan误差方差分解.json").write_text(
        json.dumps(decomposition, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    baseline_contrasts = {}
    for weighting in ["颗粒中位数", "来源等权"]:
        for name, value in calculate_contrasts(prepared["pred"], prepared, weighting).items():
            baseline_contrasts[(weighting, name)] = value

    scenario_names = ["颗粒独立误差", "来源分层误差"]
    weightings = ["颗粒中位数", "来源等权"]
    contrasts = []
    curve_store = {
        (scenario, weighting): np.empty((iterations, len(prepared["starts"])), dtype=float)
        for scenario in scenario_names for weighting in weightings
    }

    # 为两个情景使用分开的、可复现的随机数流。
    seed_sequence = np.random.SeedSequence(seed)
    rng_independent, rng_hierarchical = [np.random.default_rng(s) for s in seed_sequence.spawn(2)]

    for iteration in range(iterations):
        errors = {
            "颗粒独立误差": independent_error(
                rng_independent, centered_residuals, len(prepared["pred"])
            ),
            "来源分层误差": hierarchical_error(
                rng_hierarchical, prepared, body_offsets, within_pools
            ),
        }
        for scenario, error in errors.items():
            noisy = prepared["pred"] + error
            for weighting in weightings:
                if weighting == "颗粒中位数":
                    curve = particle_curve(noisy, prepared["bin_rows"])
                else:
                    curve = source_equal_curve(noisy, prepared)
                curve_store[(scenario, weighting)][iteration] = curve
                row = {
                    "误差情景": scenario,
                    "聚合方法": weighting,
                    "迭代": iteration + 1,
                }
                row.update(calculate_contrasts(noisy, prepared, weighting))
                contrasts.append(row)

        if (iteration + 1) % max(1, iterations // 10) == 0:
            print(f"模拟进度：{iteration + 1}/{iterations}")

    contrast_df = pd.DataFrame(contrasts)
    contrast_df.to_csv(out / "03_四情景关键对比_全部迭代.csv", index=False, encoding="utf-8-sig")
    direction = direction_summary(contrast_df, baseline_contrasts)
    direction.to_csv(out / "04_四情景方向保留概率.csv", index=False, encoding="utf-8-sig")

    baseline_curves = {
        "颗粒中位数": particle_curve(prepared["pred"], prepared["bin_rows"]),
        "来源等权": source_equal_curve(prepared["pred"], prepared),
    }
    curve_rows = []
    for scenario in scenario_names:
        for weighting in weightings:
            curves = curve_store[(scenario, weighting)]
            for bin_i, start in enumerate(prepared["starts"]):
                curve_rows.append({
                    "误差情景": scenario,
                    "聚合方法": weighting,
                    "年龄窗起点_Ma": int(start),
                    "年龄中心_Ma": int(start + 50),
                    "无噪声基线": baseline_curves[weighting][bin_i],
                    "模拟P2.5": float(np.quantile(curves[:, bin_i], .025)),
                    "模拟中位数": float(np.median(curves[:, bin_i])),
                    "模拟P97.5": float(np.quantile(curves[:, bin_i], .975)),
                })
    curve_summary = pd.DataFrame(curve_rows)
    curve_summary.to_csv(out / "05_四情景100Ma曲线汇总.csv", index=False, encoding="utf-8-sig")
    return prepared, direction, curve_summary, body_stats, decomposition


def make_figures(direction, curve_summary, body_stats, out):
    setup_plot()
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    strict = direction[
        direction["误差情景"].eq("来源分层误差") & direction["聚合方法"].eq("来源等权")
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(strict["结论"], strict["方向保留概率"], color="#3978a8")
    ax.axhline(.95, color="crimson", linestyle="--", label="95%")
    ax.axhline(.80, color="#dd6b20", linestyle=":", label="80%")
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("方向保留概率")
    ax.set_title("严格情景：来源分层误差＋来源等权")
    ax.tick_params(axis="x", rotation=18)
    ax.legend()
    savefig(fig, fig_dir / "01_严格情景方向保留概率.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, weighting in zip(axes, ["颗粒中位数", "来源等权"]):
        part = curve_summary[
            curve_summary["误差情景"].eq("来源分层误差")
            & curve_summary["聚合方法"].eq(weighting)
        ]
        x = part["年龄中心_Ma"].to_numpy(float)
        ax.fill_between(
            x,
            part["模拟P2.5"].to_numpy(float),
            part["模拟P97.5"].to_numpy(float),
            alpha=.22,
            color="#4c78a8",
            label="95%模拟区间",
        )
        ax.plot(x, part["无噪声基线"], color="black", linewidth=1.8, label="无噪声基线")
        ax.plot(x, part["模拟中位数"], color="#4c78a8", linewidth=1.2, label="模拟中位数")
        ax.set_ylabel("预测 ΔFMQ")
        ax.set_title(weighting)
        ax.legend(loc="best")
    axes[-1].set_xlabel("年龄（Ma，向右更古老）")
    fig.suptitle("Kohistan来源分层误差下的100 Ma全球曲线")
    savefig(fig, fig_dir / "02_分层误差曲线包络.png")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot = direction.copy()
    plot["情景"] = plot["误差情景"] + "＋" + plot["聚合方法"]
    conclusions = [x[0] for x in CONTRASTS]
    scenarios = plot["情景"].drop_duplicates().tolist()
    x = np.arange(len(conclusions))
    width = .18
    for i, scenario in enumerate(scenarios):
        part = plot[plot["情景"].eq(scenario)].set_index("结论").reindex(conclusions)
        ax.bar(x + (i - 1.5) * width, part["方向保留概率"], width=width, label=scenario)
    ax.axhline(.95, color="crimson", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(conclusions, rotation=18)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("方向保留概率")
    ax.set_title("颗粒独立与来源分层误差对比")
    ax.legend(fontsize=8, ncol=2)
    savefig(fig, fig_dir / "03_四情景方向概率对比.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4c78a8" if x >= 0 else "#e45756" for x in body_stats["平均残差_预测减真值"]]
    ax.bar(body_stats["岩体"], body_stats["平均残差_预测减真值"], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("平均残差（预测－真值，ΔFMQ）")
    ax.set_title("Kohistan五岩体的系统偏差")
    ax.tick_params(axis="x", rotation=18)
    savefig(fig, fig_dir / "04_Kohistan分岩体偏差.png")


def write_report(global_df, direction, body_stats, decomposition, iterations, seed, out):
    strict = direction[
        direction["误差情景"].eq("来源分层误差") & direction["聚合方法"].eq("来源等权")
    ][[
        "结论", "预期方向", "无噪声基线差值", "模拟P2.5", "模拟P97.5",
        "方向保留概率", "稳健性判断",
    ]]
    comparison = direction[["误差情景", "聚合方法", "结论", "方向保留概率", "稳健性判断"]]
    report = f"""# Kohistan分层误差敏感性检验报告

## 检验目的

旧模拟把113个Kohistan残差独立添加到每颗全球锆石。本次新增严格情景：
同一个全球来源的全部锆石共享一个由Kohistan五岩体估计的来源级偏差，
每颗锆石再叠加相应岩体内部的随机残差。

本检验只扰动已冻结预测，不训练、不调参，也不改写冻结数据。

## 运行信息

- 全球冻结数据：{len(global_df):,}颗锆石，{global_df[SOURCE_COL].nunique()}个来源
- Kohistan：{decomposition['Kohistan颗粒数']}颗锆石，{decomposition['Kohistan岩体数']}个岩体
- Kohistan总体MSE：{decomposition['总体MSE']:.3f}；RMSE：{decomposition['总体RMSE']:.3f} ΔFMQ
- 岩体均值间方差：{decomposition['岩体均值间方差']:.3f}
- 平均岩体内方差：{decomposition['平均岩体内方差']:.3f}
- 描述性ICC：{decomposition['岩体内相关近似_ICC']:.3f}
- 模拟次数：{iterations:,}；随机种子：{seed}

## 最严格情景结果

最严格情景为“来源分层误差＋来源等权”。

{markdown_table(strict)}

## 四种计算情景对比

{markdown_table(comparison)}

## Kohistan分岩体误差

{markdown_table(body_stats)}

## 使用边界

1. 方向保留概率是误差压力测试结果，不是趋势为真的概率，也不是p值。
2. 五个Kohistan岩体不足以精确代表全部全球地质背景；来源分层模拟应解释为保守压力测试。
3. 若严格情景仍保留方向，可表述为趋势对Kohistan所揭示的岩体级系统误差具有稳健性。
4. 若严格情景概率下降，不代表模型失效，而是该趋势需要降低措辞并结合独立地质证据。
"""
    (out / "06_Kohistan分层误差检验报告.md").write_text(report, encoding="utf-8")


def main():
    args = parse_args()
    if args.iterations < 100:
        raise ValueError("正式分析建议至少运行100次；默认值为2000次")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    global_df, residual_df = load_inputs(
        Path(args.global_data).resolve(), Path(args.kohistan_residuals).resolve()
    )
    print("数据核验通过：10879颗全球锆石，113颗Kohistan锆石，5个岩体")
    prepared, direction, curve_summary, body_stats, decomposition = run_simulation(
        global_df, residual_df, args.iterations, args.seed, out
    )
    make_figures(direction, curve_summary, body_stats, out)
    write_report(
        global_df, direction, body_stats, decomposition,
        args.iterations, args.seed, out,
    )
    manifest = {
        "status": "COMPLETED",
        "training_performed": False,
        "input_rows": int(len(global_df)),
        "sources": int(global_df[SOURCE_COL].nunique()),
        "kohistan_rows": int(len(residual_df)),
        "kohistan_bodies": int(residual_df[BODY_COL].nunique()),
        "iterations": int(args.iterations),
        "seed": int(args.seed),
        "primary_scenario": "来源分层误差＋来源等权",
    }
    (out / "07_运行清单.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n完成。结果保存在：{out}")
    print("请把整个“Kohistan分层误差结果”文件夹压缩后发回分析。")


if __name__ == "__main__":
    main()
