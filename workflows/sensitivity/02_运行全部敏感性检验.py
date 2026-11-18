#!/usr/bin/env python3
"""论文级敏感性检验：只读取冻结预测与冻结模型，不训练、不调参。"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent
FEATURES = ["Ce", "Nd", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "U", "Hf", "Ti"]
CN = {x: f"{x}（{y}）" for x, y in zip(FEATURES, "铈钕铕钆铽镝钬铒铥镱镥铀铪钛")}


def parse_args():
    p = argparse.ArgumentParser(description="运行最终论文数据全部敏感性检验（不会训练模型）")
    p.add_argument("--data", default=str(ROOT / "frozen/01_论文最终逐颗锆石数据_10879条_FROZEN.csv"))
    p.add_argument("--kohistan", default=str(ROOT / "inputs/Kohistan_113颗_五岩体外部验证集.csv"))
    p.add_argument("--models", default=str(ROOT / "models"))
    p.add_argument("--output", default=str(ROOT / "sensitivity_results"))
    p.add_argument("--iterations", type=int, default=2000, help="随机删除与噪声模拟次数")
    p.add_argument("--seed", type=int, default=20261118)
    return p.parse_args()


def setup_plot():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 130,
        "savefig.dpi": 220,
    })


def savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    """不依赖tabulate的简易Markdown表。"""
    q = df.copy()
    for c in q.columns:
        q[c] = q[c].map(lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x))
    head = "| " + " | ".join(map(str, q.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(q.columns)) + "|"
    rows = ["| " + " | ".join(row) + " |" for row in q.astype(str).to_numpy()]
    return "\n".join([head, sep] + rows)


def validate_data(d: pd.DataFrame):
    required = [
        "GLOBAL_RECORD_ID", "SOURCE_ID", "REFERENCE_ID", "AGE_MA_CLEAN",
        "N_FEATURES_MEASURED", "DELTA_FMQ_MLP", "DELTA_FMQ_EXTRATREES",
        "DELTA_FMQ_ENSEMBLE", "MODEL_ABS_DIFF",
    ]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise RuntimeError(f"冻结数据缺少字段：{missing}")
    if len(d) != 10879 or d.GLOBAL_RECORD_ID.nunique() != 10879:
        raise RuntimeError("冻结数据记录数或唯一ID不符合10879条要求")
    if d[required].isna().any().any():
        bad = d[required].isna().sum()
        raise RuntimeError(f"冻结数据关键字段仍有空值：{bad[bad > 0].to_dict()}")


def add_bins(d: pd.DataFrame, width: int) -> pd.DataFrame:
    x = d.copy()
    x["BIN_START"] = (np.floor(x.AGE_MA_CLEAN / width) * width).astype(int)
    x["BIN_CENTER"] = x.BIN_START + width / 2
    return x


def bin_summary(d: pd.DataFrame, width: int) -> pd.DataFrame:
    x = add_bins(d, width)
    rows = []
    for start, g in x.groupby("BIN_START", sort=True):
        counts = g.SOURCE_ID.value_counts()
        src_med = g.groupby("SOURCE_ID").DELTA_FMQ_ENSEMBLE.median()
        rows.append({
            "窗口宽度_Ma": width,
            "年龄窗起点_Ma": int(start),
            "年龄窗终点_Ma": int(start + width),
            "年龄中心_Ma": float(start + width / 2),
            "颗粒数": int(len(g)),
            "独立来源数": int(g.SOURCE_ID.nunique()),
            "独立文献数": int(g.REFERENCE_ID.nunique()),
            "最大来源占比": float(counts.iloc[0] / len(g)),
            "14元素完整数": int((g.N_FEATURES_MEASURED == 14).sum()),
            "缺1至2元素比例": float((g.N_FEATURES_MEASURED < 14).mean()),
            "MLP_中位数": float(g.DELTA_FMQ_MLP.median()),
            "ExtraTrees_中位数": float(g.DELTA_FMQ_EXTRATREES.median()),
            "双模型颗粒中位数": float(g.DELTA_FMQ_ENSEMBLE.median()),
            "双模型Q25": float(g.DELTA_FMQ_ENSEMBLE.quantile(.25)),
            "双模型Q75": float(g.DELTA_FMQ_ENSEMBLE.quantile(.75)),
            "来源等权中位数": float(src_med.median()),
            "模型差异中位数": float(g.MODEL_ABS_DIFF.median()),
        })
    return pd.DataFrame(rows)


def source_bin_matrix(d: pd.DataFrame, width: int = 100):
    x = add_bins(d, width)
    sources = np.array(sorted(x.SOURCE_ID.astype(str).unique()))
    bins = np.array(sorted(x.BIN_START.unique()))
    table = x.groupby(["SOURCE_ID", "BIN_START"]).DELTA_FMQ_ENSEMBLE.median().unstack()
    table = table.reindex(index=sources, columns=bins)
    return sources, bins, table.to_numpy(float)


def leave_one_source_out(d: pd.DataFrame, out: Path):
    sources, bins, matrix = source_bin_matrix(d, 100)
    baseline = np.nanmedian(matrix, axis=0)
    detail = []
    curves = []
    for i, source in enumerate(sources):
        m = matrix.copy()
        m[i, :] = np.nan
        curve = np.nanmedian(m, axis=0)
        curves.append(curve)
        for j, b in enumerate(bins):
            detail.append({
                "删除来源": source, "年龄窗起点_Ma": int(b), "年龄中心_Ma": float(b + 50),
                "基线来源等权中位数": baseline[j], "删除后来源等权中位数": curve[j],
                "变化量": curve[j] - baseline[j],
            })
    detail = pd.DataFrame(detail)
    detail.to_csv(out / "04_LOSO逐来源逐窗口.csv", index=False, encoding="utf-8-sig")
    curves = np.asarray(curves)
    summary = pd.DataFrame({
        "年龄窗起点_Ma": bins,
        "年龄中心_Ma": bins + 50,
        "基线来源等权中位数": baseline,
        "LOSO最小值": np.nanmin(curves, axis=0),
        "LOSO最大值": np.nanmax(curves, axis=0),
        "LOSO_P2.5": np.nanquantile(curves, .025, axis=0),
        "LOSO_P97.5": np.nanquantile(curves, .975, axis=0),
        "最大绝对变化": np.nanmax(np.abs(curves - baseline), axis=0),
    })
    summary.to_csv(out / "05_LOSO汇总.csv", index=False, encoding="utf-8-sig")
    return summary


def random_source_deletion(d: pd.DataFrame, out: Path, iterations: int, rng):
    sources, bins, matrix = source_bin_matrix(d, 100)
    baseline = np.nanmedian(matrix, axis=0)
    summaries = []
    iteration_rows = []
    saved_curves = {}
    for fraction in [.10, .20, .30]:
        n_remove = max(1, int(round(len(sources) * fraction)))
        curves = np.empty((iterations, len(bins)))
        for k in range(iterations):
            remove = rng.choice(len(sources), size=n_remove, replace=False)
            keep = np.ones(len(sources), dtype=bool)
            keep[remove] = False
            curves[k] = np.nanmedian(matrix[keep], axis=0)
            valid = np.isfinite(curves[k]) & np.isfinite(baseline)
            corr = np.corrcoef(curves[k, valid], baseline[valid])[0, 1]
            iteration_rows.append({
                "删除比例": fraction, "迭代": k + 1, "删除来源数": n_remove,
                "与基线相关系数": corr,
                "最大绝对偏移": float(np.nanmax(np.abs(curves[k] - baseline))),
                "平均绝对偏移": float(np.nanmean(np.abs(curves[k] - baseline))),
            })
        saved_curves[fraction] = curves
        for j, b in enumerate(bins):
            summaries.append({
                "删除比例": fraction, "年龄窗起点_Ma": int(b), "年龄中心_Ma": float(b + 50),
                "基线来源等权中位数": baseline[j],
                "随机删除_P2.5": np.nanquantile(curves[:, j], .025),
                "随机删除_中位数": np.nanmedian(curves[:, j]),
                "随机删除_P97.5": np.nanquantile(curves[:, j], .975),
            })
    pd.DataFrame(iteration_rows).to_csv(out / "06_随机删除来源_迭代指标.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "07_随机删除来源_逐窗口汇总.csv", index=False, encoding="utf-8-sig")
    return summary, pd.DataFrame(iteration_rows)


def load_old_models(folder: Path):
    if sklearn.__version__ != "1.3.2":
        raise RuntimeError(f"冻结模型要求scikit-learn==1.3.2，当前为{sklearn.__version__}")
    files = list(folder.glob("*.joblib"))
    mlp_file = next((p for p in files if "MLP" in p.name.upper()), None)
    et_file = next((p for p in files if "EXTRA" in p.name.upper()), None)
    if mlp_file is None or et_file is None:
        raise FileNotFoundError("models中必须同时存在MLP和ExtraTrees模型")
    mlp = joblib.load(mlp_file)["bundle"]
    et = joblib.load(et_file)["bundle"]
    for name, bundle in [("MLP", mlp), ("ExtraTrees", et)]:
        model_features = [re.split(r"[（(]", str(x), maxsplit=1)[0] for x in bundle["features"]]
        if model_features != FEATURES:
            raise RuntimeError(f"{name}特征顺序不一致：{model_features}")
    return mlp, et


def kohistan_residuals(path: Path, models, out: Path):
    k = pd.read_csv(path, low_memory=False)
    X = pd.DataFrame({f: pd.to_numeric(k[CN[f]], errors="coerce") for f in FEATURES})
    X = np.log10(X.where(X > 0)).to_numpy()
    mlp, et = models
    xm = mlp["imputer"].transform(X)
    pm = np.mean([m.predict(mlp["scaler"].transform(xm)) for m in mlp["models"]], axis=0)
    xe = et["imputer"].transform(X)
    pe = et["model"].predict(xe)
    truth = pd.to_numeric(k["独立约束_ΔFMQ"], errors="coerce").to_numpy()
    k["DELTA_FMQ_MLP"] = pm
    k["DELTA_FMQ_EXTRATREES"] = pe
    k["DELTA_FMQ_ENSEMBLE"] = (pm + pe) / 2
    k["ENSEMBLE_RESIDUAL_PRED_MINUS_TRUE"] = k.DELTA_FMQ_ENSEMBLE - truth
    k.to_csv(out / "08_Kohistan真实残差.csv", index=False, encoding="utf-8-sig")
    return k.ENSEMBLE_RESIDUAL_PRED_MINUS_TRUE.to_numpy(float)


def window_median(values, ages, lo, hi):
    return float(np.median(values[(ages >= lo) & (ages < hi)]))


def kohistan_noise_simulation(d: pd.DataFrame, residuals, out: Path, iterations: int, rng):
    ages = d.AGE_MA_CLEAN.to_numpy(float)
    pred = d.DELTA_FMQ_ENSEMBLE.to_numpy(float)
    residuals = np.asarray(residuals, float)
    centered = residuals - residuals.mean()
    starts = np.arange(0, 4000, 100)
    masks = [(ages >= b) & (ages < b + 100) for b in starts]
    curves = np.empty((iterations, len(starts)))
    contrasts = []
    for i in range(iterations):
        noisy = pred + rng.choice(centered, size=len(pred), replace=True)
        curves[i] = [np.median(noisy[m]) for m in masks]
        ancient = window_median(noisy, ages, 3000, 3300) - window_median(noisy, ages, 3500, 4001)
        late_arch = window_median(noisy, ages, 2500, 3000) - window_median(noisy, ages, 1800, 2500)
        mid = window_median(noisy, ages, 800, 1800)
        shoulders = (window_median(noisy, ages, 400, 800) + window_median(noisy, ages, 1800, 2500)) / 2
        recovery = window_median(noisy, ages, 0, 400) - mid
        contrasts.append({
            "迭代": i + 1, "古老地球氧化上升": ancient, "晚太古代高值": late_arch,
            "中元古代低值_相对两侧": mid - shoulders, "显生宙恢复": recovery,
        })
    baseline = np.array([np.median(pred[m]) for m in masks])
    curve_summary = pd.DataFrame({
        "年龄窗起点_Ma": starts, "年龄中心_Ma": starts + 50,
        "原始颗粒中位数": baseline,
        "噪声模拟_P2.5": np.quantile(curves, .025, axis=0),
        "噪声模拟_中位数": np.median(curves, axis=0),
        "噪声模拟_P97.5": np.quantile(curves, .975, axis=0),
    })
    curve_summary.to_csv(out / "09_Kohistan噪声模拟_逐窗口汇总.csv", index=False, encoding="utf-8-sig")
    contrast_df = pd.DataFrame(contrasts)
    contrast_df.to_csv(out / "10_Kohistan噪声模拟_关键对比.csv", index=False, encoding="utf-8-sig")
    direction = pd.DataFrame([
        {"结论": "古老地球氧化上升", "预期方向": ">0", "方向保留概率": float((contrast_df["古老地球氧化上升"] > 0).mean())},
        {"结论": "晚太古代高值", "预期方向": ">0", "方向保留概率": float((contrast_df["晚太古代高值"] > 0).mean())},
        {"结论": "中元古代低值", "预期方向": "<0", "方向保留概率": float((contrast_df["中元古代低值_相对两侧"] < 0).mean())},
        {"结论": "显生宙恢复", "预期方向": ">0", "方向保留概率": float((contrast_df["显生宙恢复"] > 0).mean())},
    ])
    direction.to_csv(out / "11_Kohistan噪声模拟_方向保留概率.csv", index=False, encoding="utf-8-sig")
    residual_stats = {
        "n": int(len(residuals)), "bias": float(residuals.mean()),
        "mae": float(np.mean(np.abs(residuals))), "mse": float(np.mean(residuals ** 2)),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))), "noise_used": "centered empirical residual bootstrap",
    }
    return curve_summary, direction, residual_stats


def complete_element_sensitivity(d: pd.DataFrame, out: Path):
    full = d[d.N_FEATURES_MEASURED == 14]
    all_t = bin_summary(d, 100)
    full_t = bin_summary(full, 100)
    q = all_t[["年龄窗起点_Ma", "年龄中心_Ma", "双模型颗粒中位数", "来源等权中位数"]].merge(
        full_t[["年龄窗起点_Ma", "双模型颗粒中位数", "来源等权中位数", "颗粒数"]],
        on="年龄窗起点_Ma", suffixes=("_全部10879", "_完整14元素"), how="left")
    q["颗粒中位数差_完整减全部"] = q["双模型颗粒中位数_完整14元素"] - q["双模型颗粒中位数_全部10879"]
    q["来源等权差_完整减全部"] = q["来源等权中位数_完整14元素"] - q["来源等权中位数_全部10879"]
    q.to_csv(out / "12_完整14元素敏感性.csv", index=False, encoding="utf-8-sig")
    return q


def make_figures(window_tables, loso, random_summary, random_iter, noise_curve, direction, complete, d, fig_dir):
    setup_plot()
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for width, t in window_tables.items():
        ax.plot(t["年龄中心_Ma"], t["来源等权中位数"], marker="o", ms=3, label=f"{width} Ma")
    ax.set(xlabel="年龄（Ma，向右更古老）", ylabel="来源等权预测 ΔFMQ", title="50/100/150 Ma时间窗敏感性")
    ax.legend(); ax.grid(alpha=.2); savefig(fig, fig_dir / "01_时间窗敏感性.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = loso["年龄中心_Ma"]
    ax.fill_between(x, loso["LOSO最小值"], loso["LOSO最大值"], alpha=.18, color="gray", label="删除任一来源范围")
    ax.plot(x, loso["基线来源等权中位数"], color="black", lw=2, label="基线")
    ax.set(xlabel="年龄（Ma）", ylabel="来源等权 ΔFMQ", title="Leave-one-source-out敏感性")
    ax.legend(); savefig(fig, fig_dir / "02_LOSO包络.png")

    rs = random_summary[random_summary["删除比例"].eq(.20)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].fill_between(rs["年龄中心_Ma"], rs["随机删除_P2.5"], rs["随机删除_P97.5"], alpha=.22)
    axes[0].plot(rs["年龄中心_Ma"], rs["基线来源等权中位数"], color="black")
    axes[0].set(title="随机删除20%来源", xlabel="年龄（Ma）", ylabel="来源等权 ΔFMQ")
    for frac, g in random_iter.groupby("删除比例"):
        axes[1].hist(g["与基线相关系数"], bins=35, alpha=.5, label=f"删除{frac:.0%}")
    axes[1].set(title="随机删除后与基线曲线相关性", xlabel="相关系数", ylabel="次数"); axes[1].legend()
    savefig(fig, fig_dir / "03_随机删除来源.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].fill_between(noise_curve["年龄中心_Ma"], noise_curve["噪声模拟_P2.5"], noise_curve["噪声模拟_P97.5"], alpha=.2)
    axes[0].plot(noise_curve["年龄中心_Ma"], noise_curve["原始颗粒中位数"], color="black")
    axes[0].set(title="叠加Kohistan真实残差", xlabel="年龄（Ma）", ylabel="预测 ΔFMQ")
    axes[1].bar(direction["结论"], direction["方向保留概率"], color="#3978a8")
    axes[1].axhline(.95, color="crimson", ls="--"); axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=25); axes[1].set(title="关键方向保留概率", ylabel="概率")
    savefig(fig, fig_dir / "04_Kohistan噪声模拟.png")

    t = window_tables[100]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(t["年龄中心_Ma"], t["颗粒数"]); axes[0].set_ylabel("颗粒数")
    axes[1].plot(t["年龄中心_Ma"], t["独立来源数"], color="#2f855a"); axes[1].set_ylabel("独立来源数")
    axes[2].plot(t["年龄中心_Ma"], t["最大来源占比"], color="#dd6b20"); axes[2].set_ylabel("最大来源占比"); axes[2].set_xlabel("年龄（Ma）")
    fig.suptitle("每100 Ma数据支撑度"); savefig(fig, fig_dir / "05_每窗颗粒与来源.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(complete["年龄中心_Ma"], complete["双模型颗粒中位数_全部10879"], label="全部10879条")
    ax.plot(complete["年龄中心_Ma"], complete["双模型颗粒中位数_完整14元素"], label="完整14元素")
    ax.set(title="缺失元素敏感性", xlabel="年龄（Ma）", ylabel="预测 ΔFMQ"); ax.legend()
    savefig(fig, fig_dir / "06_完整14元素敏感性.png")

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(d.DELTA_FMQ_MLP, d.DELTA_FMQ_EXTRATREES, s=5, alpha=.2)
    lo = min(d.DELTA_FMQ_MLP.min(), d.DELTA_FMQ_EXTRATREES.min())
    hi = max(d.DELTA_FMQ_MLP.max(), d.DELTA_FMQ_EXTRATREES.max())
    ax.plot([lo, hi], [lo, hi], "k--"); ax.set(xlabel="MLP ΔFMQ", ylabel="ExtraTrees ΔFMQ", title="两冻结模型一致性")
    savefig(fig, fig_dir / "07_模型一致性.png")


def write_report(out: Path, d, window_tables, loso, random_iter, residual_stats, direction, complete, iterations, seed):
    t100 = window_tables[100]
    weak = t100.nsmallest(5, "独立来源数")[["年龄窗起点_Ma", "颗粒数", "独立来源数", "最大来源占比"]]
    report = f"""# 最终论文数据敏感性检验报告

## 运行信息

- 冻结数据：{len(d):,}条，{d.SOURCE_ID.nunique()}个来源，{d.REFERENCE_ID.nunique()}篇文献
- 模型：旧版MLP与ExtraTrees冻结模型；本程序未训练、未调参
- 随机模拟：{iterations:,}次；随机种子：{seed}
- Kohistan残差：n={residual_stats['n']}，MSE={residual_stats['mse']:.3f}，RMSE={residual_stats['rmse']:.3f}，bias={residual_stats['bias']:.3f}

## 已完成检验

1. Leave-one-source-out：逐一删除55个来源并重算100 Ma来源等权曲线。
2. 随机来源删除：分别删除10%、20%、30%来源。
3. 50、100、150 Ma时间窗敏感性。
4. 使用Kohistan真实残差经验分布进行噪声模拟。
5. 每窗颗粒数、来源数、文献数和最大来源占比。
6. 完整14元素数据敏感性。

## 自动摘要

- LOSO所有窗口最大绝对偏移的最大值：{loso['最大绝对变化'].max():.3f} ΔFMQ。
- 随机删除20%来源后，与基线曲线相关系数中位数：{random_iter.loc[random_iter['删除比例'].eq(.20), '与基线相关系数'].median():.3f}。
- 完整14元素与全部数据的颗粒中位曲线：窗口差异中位绝对值为{complete['颗粒中位数差_完整减全部'].abs().median():.3f} ΔFMQ，最大值为{complete['颗粒中位数差_完整减全部'].abs().max():.3f} ΔFMQ。

### Kohistan噪声下方向保留概率

{markdown_table(direction)}

### 独立来源最少的5个100 Ma窗口

{markdown_table(weak)}

所有逐次结果、汇总表和补充图均保存在本目录。方向概率是模拟稳定性指标，不是p值。
"""
    (out / "敏感性检验报告.md").write_text(report, encoding="utf-8")


def main():
    a = parse_args()
    out = Path(a.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    d = pd.read_csv(a.data, low_memory=False)
    validate_data(d)
    rng = np.random.default_rng(a.seed)

    window_tables = {}
    for width in [50, 100, 150]:
        t = bin_summary(d, width)
        t.to_csv(out / f"0{width // 50}_时间窗_{width}Ma.csv", index=False, encoding="utf-8-sig")
        window_tables[width] = t

    loso = leave_one_source_out(d, out)
    random_summary, random_iter = random_source_deletion(d, out, a.iterations, rng)
    models = load_old_models(Path(a.models))
    residuals = kohistan_residuals(Path(a.kohistan), models, out)
    noise_curve, direction, residual_stats = kohistan_noise_simulation(d, residuals, out, a.iterations, rng)
    complete = complete_element_sensitivity(d, out)
    make_figures(window_tables, loso, random_summary, random_iter, noise_curve, direction, complete, d, fig_dir)
    write_report(out, d, window_tables, loso, random_iter, residual_stats, direction, complete, a.iterations, a.seed)

    manifest = {
        "status": "COMPLETED", "training_performed": False,
        "input_rows": len(d), "sources": int(d.SOURCE_ID.nunique()),
        "references": int(d.REFERENCE_ID.nunique()), "iterations": a.iterations,
        "seed": a.seed, "sklearn": sklearn.__version__,
    }
    (out / "运行清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"全部敏感性检验完成：{out}")


if __name__ == "__main__":
    main()
