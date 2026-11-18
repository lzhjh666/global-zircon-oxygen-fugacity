# -*- coding: utf-8 -*-
"""外部验证和ExtraTrees内部正则化共用函数。"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import mean_squared_error, r2_score


ALL_FEATURES = [
    "Ce（铈）", "Nd（钕）", "Sm（钐）", "Eu（铕）", "Gd（钆）", "Tb（铽）",
    "Dy（镝）", "Ho（钬）", "Er（铒）", "Tm（铥）", "Yb（镱）", "Lu（镥）",
    "Th（钍）", "U（铀）", "Hf（铪）", "Ti（钛）", "Y（钇）",
]
TARGET = "目标氧逸度（log fO₂−FMQ）"
GROUP = "最终分组编号"
SITE = "site"
SD = "目标值标准差（FMQ）"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path, sheet_name=0) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def load_locked_bundle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"没有找到模型文件：{path}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = joblib.load(path)
    mismatch = [item for item in caught if issubclass(item.category, InconsistentVersionWarning)]
    if mismatch:
        messages = "\n".join(str(item.message) for item in mismatch[:3])
        raise RuntimeError(
            "模型文件与当前scikit-learn版本不一致。为避免产生不可信的外部验证数值，程序已经停止。\n"
            "请回到生成该模型时使用的Python环境再运行。\n" + messages
        )
    if not isinstance(obj, dict) or "bundle" not in obj:
        raise ValueError(f"模型文件结构不符合本项目锁定模型格式：{path.name}")
    bundle = obj["bundle"]
    required = {"kind", "features", "imputer"}
    if not required.issubset(bundle):
        raise ValueError(f"模型缺少字段：{sorted(required - set(bundle))}")
    return bundle


def prepare_features(frame: pd.DataFrame, features: list[str], name: str) -> pd.DataFrame:
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise ValueError(f"{name}缺少模型需要的元素列：{'、'.join(missing)}")
    out = frame.copy()
    out[features] = out[features].apply(pd.to_numeric, errors="coerce")
    out[features] = out[features].mask(out[features] <= 0)
    all_missing = [feature for feature in features if out[feature].isna().all()]
    if all_missing:
        raise ValueError(f"{name}以下元素整列缺失：{'、'.join(all_missing)}")
    return out


def log_x(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(frame[features].astype(float).to_numpy())


def predict_bundle(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    features = list(bundle["features"])
    prepared = prepare_features(frame, features, "待预测数据")
    x = bundle["imputer"].transform(log_x(prepared, features))
    if bundle["kind"] == "MLP_64x32":
        if "scaler" not in bundle or "models" not in bundle:
            raise ValueError("MLP锁定模型缺少scaler或models。")
        x = bundle["scaler"].transform(x)
        return np.mean([model.predict(x) for model in bundle["models"]], axis=0)
    if "model" not in bundle:
        raise ValueError("树模型锁定文件缺少model。")
    return bundle["model"].predict(x)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y = np.asarray(y_true, float)
    pred = np.asarray(y_pred, float)
    valid = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[valid], pred[valid]
    if len(y) == 0:
        raise ValueError("没有可评分的真实值与预测值。")
    mse = float(mean_squared_error(y, pred))
    return {
        "R2": float(r2_score(y, pred)) if len(y) >= 2 and np.unique(y).size >= 2 else np.nan,
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(pred - y))),
        "平均偏差_预测减真实": float(np.mean(pred - y)),
        "样本数": int(len(y)),
    }


def group_predictions(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    if GROUP not in frame or TARGET not in frame:
        raise ValueError("组级评分需要最终分组编号和目标氧逸度列。")
    work = frame[[GROUP, TARGET]].copy()
    work["预测ΔFMQ"] = np.asarray(prediction, float)
    result = work.groupby(GROUP, as_index=False).agg(
        真实ΔFMQ=(TARGET, "first"),
        预测ΔFMQ=("预测ΔFMQ", "median"),
        锆石数=("预测ΔFMQ", "size"),
    )
    result["残差_预测减真实"] = result["预测ΔFMQ"] - result["真实ΔFMQ"]
    result["绝对误差"] = result["残差_预测减真实"].abs()
    return result


def dataframe_text(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "（无数据）"
    return frame.to_string(index=False, float_format=lambda value: f"{value:.6f}")
