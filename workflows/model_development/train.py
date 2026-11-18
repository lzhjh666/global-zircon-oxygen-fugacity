# -*- coding: utf-8 -*-
"""锆石氧逸度：逐元素向后消融、整地点训练、锁定地点测试与 JIM 外部验证。

正式评价只使用地点级划分。所有插补、标准化和早停均只在训练地点内完成。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
FEATURES = [
    'Ce（铈）', 'Nd（钕）', 'Sm（钐）', 'Eu（铕）', 'Gd（钆）', 'Tb（铽）',
    'Dy（镝）', 'Ho（钬）', 'Er（铒）', 'Tm（铥）', 'Yb（镱）', 'Lu（镥）',
    'Th（钍）', 'U（铀）', 'Hf（铪）', 'Ti（钛）', 'Y（钇）'
]
SYMBOL = {x: x.split('（')[0] for x in FEATURES}
TARGET = '目标氧逸度（log fO₂−FMQ）'
GROUP = '最终分组编号'
SITE = 'site'
SD = '目标值标准差（FMQ）'
SEED = 81001
EXCLUDE_REASONS = {
    'Mt St Helens, Yn tephra, 3510 yr BP | ΔFMQ=2.75':
        '阶段配对存疑：7个锆石微区共用Yn事件标签，缺少逐微区年龄和同阶段Fe-Ti氧化物对应。',
    'Kimberlites, Kaapvaal, Southern Africa | ΔFMQ=-2.810':
        '应用域与标签对应问题：27颗地幔来源金伯利岩巨晶来自多个矿山/岩管，却共用单一捕虏体约束；不属于大陆地壳碎屑锆石主应用域。',
}
FEATURE_ALIASES = {
    f: [f, SYMBOL[f], f'{SYMBOL[f]}_ppm', f'{SYMBOL[f]} ppm', f'{SYMBOL[f]}(ppm)', f'{SYMBOL[f]}（ppm）']
    for f in FEATURES
}
TARGET_ALIASES = [TARGET, 'ΔFMQ', 'dFMQ', 'deltaFMQ', 'FMQ', '真实ΔFMQ', '目标ΔFMQ', 'true']
ID_ALIASES = ['样品编号', '锆石分析编号', 'Sample_ID', 'sample_id', 'ID', 'id', 'Name', 'name']
EXTERNAL_GROUP_ALIASES = [GROUP, '外部分组', 'group', 'Group', '样品组', '地点组']


def set_plot_style() -> None:
    candidates = [
        Path('C:/Windows/Fonts/msyh.ttc'), Path('C:/Windows/Fonts/simhei.ttf'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'),
    ]
    chosen = None
    for font_path in candidates:
        if font_path.exists():
            try:
                font_manager.fontManager.addfont(str(font_path))
                chosen = font_manager.FontProperties(fname=str(font_path)).get_name()
                break
            except Exception:
                pass
    plt.rcParams['font.sans-serif'] = ([chosen] if chosen else []) + [
        'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans'
    ]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 120


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    for enc in ('utf-8-sig', 'utf-8', 'gb18030'):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_main(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = read_table(path)
    required = FEATURES + [TARGET, GROUP, SITE, SD]
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError('训练表缺少列：' + '、'.join(missing))
    d = raw.dropna(subset=[TARGET, GROUP, SITE]).copy()
    d[TARGET] = pd.to_numeric(d[TARGET], errors='coerce')
    d[SD] = pd.to_numeric(d[SD], errors='coerce')
    d[FEATURES] = d[FEATURES].apply(pd.to_numeric, errors='coerce')
    # 0 和负数不能取对数，按未检出/无效浓度处理，由训练折中位数插补。
    d[FEATURES] = d[FEATURES].mask(d[FEATURES] <= 0)
    d = d.dropna(subset=[TARGET]).reset_index(drop=True)
    if d.groupby(GROUP)[TARGET].nunique().max() != 1:
        raise ValueError('同一标签组存在多个目标值，请先核对标签。')
    if d.groupby(GROUP)[SITE].nunique().max() != 1:
        raise ValueError('同一标签组跨越多个地点，请先核对地点映射。')
    if d[FEATURES].isna().all(axis=0).any():
        bad = d[FEATURES].columns[d[FEATURES].isna().all()].tolist()
        raise ValueError('以下元素全部缺失：' + '、'.join(bad))
    excluded = d[d[GROUP].isin(EXCLUDE_REASONS)].copy()
    absent = sorted(set(EXCLUDE_REASONS) - set(excluded[GROUP]))
    if absent:
        raise ValueError('预设剔除组未找到：' + '；'.join(absent))
    excluded['剔除原因'] = excluded[GROUP].map(EXCLUDE_REASONS)
    kept = d[~d[GROUP].isin(EXCLUDE_REASONS)].copy().reset_index(drop=True)
    if len(excluded) != 34 or len(kept) != 1232 or kept[GROUP].nunique() != 36 or kept[SITE].nunique() != 25:
        raise AssertionError(
            f'数据数量与已审计版本不一致：剔除{len(excluded)}，保留{len(kept)}，'
            f'标签组{kept[GROUP].nunique()}，地点{kept[SITE].nunique()}。程序已停止。'
        )
    return raw, kept, excluded


def site_targets(d: pd.DataFrame) -> pd.Series:
    return d.groupby(SITE)[TARGET].median().sort_index()


def target_band(values: pd.Series) -> pd.Series:
    return pd.cut(values, [-np.inf, 0.5, 2.0, np.inf], labels=['低', '中', '高'], right=False)


def choose_lockbox_sites(d: pd.DataFrame, fraction: float, seed: int) -> list[str]:
    sy = site_targets(d)
    bands = target_band(sy)
    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    for band in ['低', '中', '高']:
        names = np.asarray(sorted(sy.index[bands == band]), dtype=object)
        if len(names) == 0:
            continue
        rng.shuffle(names)
        n = max(1, int(round(len(names) * fraction)))
        n = min(n, max(1, len(names) - 1))
        chosen.extend(map(str, names[:n]))
    return sorted(set(chosen))


def site_folds(d: pd.DataFrame, n_splits: int, seed: int):
    sy = site_targets(d)
    if len(sy) < n_splits:
        raise ValueError('地点数少于折数。')
    bands = target_band(sy)
    held = [set() for _ in range(n_splits)]
    rng = np.random.default_rng(seed)
    for band in ['低', '中', '高']:
        names = np.asarray(sorted(sy.index[bands == band]), dtype=object)
        rng.shuffle(names)
        for i, name in enumerate(names):
            held[i % n_splits].add(str(name))
    for fold, sites in enumerate(held, 1):
        train = d.loc[~d[SITE].isin(sites)].copy().reset_index(drop=True)
        test = d.loc[d[SITE].isin(sites)].copy().reset_index(drop=True)
        if train.empty or test.empty:
            raise ValueError('出现空折，请降低折数。')
        yield fold, sorted(sites), train, test


def log_x(d: pd.DataFrame, features: list[str]) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log10(d[features].astype(float).to_numpy())


def aggregate_groups(d: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    cols = [GROUP, SITE, TARGET]
    if SD in d.columns:
        cols.append(SD)
    z = d[cols].copy()
    z['预测值'] = np.asarray(prediction, dtype=float)
    agg = {TARGET: 'first', '预测值': 'median'}
    if SD in z.columns:
        agg[SD] = 'first'
    out = z.groupby([GROUP, SITE], as_index=False).agg(agg)
    out = out.rename(columns={TARGET: '真实值'})
    out['残差'] = out['预测值'] - out['真实值']
    out['绝对误差'] = out['残差'].abs()
    return out


def metrics_from_groups(groups: pd.DataFrame) -> dict[str, float]:
    y = groups['真实值'].to_numpy(float)
    p = groups['预测值'].to_numpy(float)
    mse = float(mean_squared_error(y, p))
    r2 = float(r2_score(y, p)) if len(y) >= 2 and np.unique(y).size >= 2 else np.nan
    return {'R2': r2, 'MSE': mse, 'RMSE': float(np.sqrt(mse))}


def balanced_indices(d: pd.DataFrame, seed: int, per_group: int = 24) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for _, idx in d.groupby(GROUP).indices.items():
        idx = np.asarray(idx, dtype=int)
        selected.extend(rng.choice(idx, size=per_group, replace=len(idx) < per_group).tolist())
    selected = np.asarray(selected, dtype=int)
    rng.shuffle(selected)
    return selected


def new_mlp(seed: int, alpha: float) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=alpha,
        batch_size=64, learning_rate_init=0.001, max_iter=1, warm_start=True,
        shuffle=True, tol=0.0, random_state=seed
    )


def train_mlp_epochs(
    d: pd.DataFrame, features: list[str], seed: int, epochs: int, alpha: float,
    imputer: SimpleImputer | None = None, scaler: StandardScaler | None = None
) -> dict:
    x = log_x(d, features)
    y = d[TARGET].to_numpy(float)
    if imputer is None:
        imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
        xi = imputer.fit_transform(x)
    else:
        xi = imputer.transform(x)
    if scaler is None:
        scaler = StandardScaler()
        xs = scaler.fit_transform(xi)
    else:
        xs = scaler.transform(xi)
    model = new_mlp(seed, alpha)
    for epoch in range(1, max(1, int(epochs)) + 1):
        idx = balanced_indices(d, seed + epoch * 7919)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.partial_fit(xs[idx], y[idx])
    return {'kind': 'MLP_64x32', 'features': list(features), 'imputer': imputer,
            'scaler': scaler, 'models': [model], 'epochs': int(epochs), 'alpha': float(alpha)}


def fit_mlp_group_early_stop(
    train: pd.DataFrame, features: list[str], seed: int, fast: bool,
    max_epochs: int | None = None, patience: int | None = None,
    initializations: int = 1, alpha: float = 0.20
) -> tuple[dict, dict]:
    max_epochs = max_epochs or (45 if fast else 420)
    patience = patience or (7 if fast else 55)
    val_sites = choose_lockbox_sites(train, 0.20, seed + 313)
    fit_d = train.loc[~train[SITE].isin(val_sites)].copy().reset_index(drop=True)
    val_d = train.loc[train[SITE].isin(val_sites)].copy().reset_index(drop=True)
    if fit_d[SITE].nunique() < 5 or val_d.empty:
        # 极小数据时仅用于流程兼容；本项目25地点不会走到这里。
        val_sites = sorted(train[SITE].unique())[-1:]
        fit_d = train.loc[~train[SITE].isin(val_sites)].copy().reset_index(drop=True)
        val_d = train.loc[train[SITE].isin(val_sites)].copy().reset_index(drop=True)

    best_epochs: list[int] = []
    best_scores: list[float] = []
    histories: list[dict] = []
    for init in range(initializations):
        s = seed + init * 100003
        imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
        scaler = StandardScaler()
        xfit = scaler.fit_transform(imputer.fit_transform(log_x(fit_d, features)))
        xval = scaler.transform(imputer.transform(log_x(val_d, features)))
        yfit = fit_d[TARGET].to_numpy(float)
        model = new_mlp(s, alpha)
        best_model = None
        best_mse = np.inf
        best_epoch = 1
        stale = 0
        init_history = []
        for epoch in range(1, max_epochs + 1):
            idx = balanced_indices(fit_d, s + epoch * 7919)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.partial_fit(xfit[idx], yfit[idx])
            if epoch < 10 and epoch % 2:
                continue
            val_pred = model.predict(xval)
            val_groups = aggregate_groups(val_d, val_pred)
            mse = metrics_from_groups(val_groups)['MSE']
            init_history.append({'epoch': epoch, 'validation_MSE': mse})
            if mse < best_mse - 1e-4:
                best_mse = mse
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                stale = 0
            else:
                stale += 1
            if epoch >= 30 and stale >= patience:
                break
        if best_model is None:
            best_model = copy.deepcopy(model)
        best_epochs.append(best_epoch)
        best_scores.append(best_mse)
        histories.append({'initialization': init + 1, 'best_epoch': best_epoch,
                          'best_validation_MSE': best_mse, 'curve': init_history})

    # 早停轮数由完全独立的训练内地点给出，然后在全部外层训练地点上重训。
    refit_models = []
    final_imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
    final_scaler = StandardScaler()
    final_xi = final_imputer.fit_transform(log_x(train, features))
    final_scaler.fit(final_xi)
    for init, epochs in enumerate(best_epochs):
        s = seed + init * 100003
        bundle = train_mlp_epochs(
            train, features, s, epochs, alpha,
            imputer=final_imputer, scaler=final_scaler
        )
        refit_models.append(bundle['models'][0])
    result = {'kind': 'MLP_64x32', 'features': list(features),
              'imputer': final_imputer, 'scaler': final_scaler,
              'models': refit_models, 'epochs': best_epochs, 'alpha': float(alpha)}
    info = {'validation_sites': val_sites, 'best_epochs': best_epochs,
            'mean_best_validation_MSE': float(np.mean(best_scores)), 'histories': histories}
    return result, info


def fit_extratrees(train: pd.DataFrame, features: list[str], seed: int, fast: bool,
                   final: bool = False) -> tuple[dict, dict]:
    imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
    x = imputer.fit_transform(log_x(train, features))
    y = train[TARGET].to_numpy(float)
    weights = 1.0 / train.groupby(GROUP)[GROUP].transform('size').to_numpy(float)
    n_trees = 80 if fast else (1200 if final else 350)
    model = ExtraTreesRegressor(
        n_estimators=n_trees, min_samples_leaf=3, max_features=0.75,
        bootstrap=False, random_state=seed, n_jobs=-1
    )
    model.fit(x, y, sample_weight=weights)
    bundle = {'kind': 'ExtraTrees', 'features': list(features), 'imputer': imputer,
              'model': model, 'n_estimators': n_trees}
    return bundle, {'n_estimators': n_trees}


def predict_bundle(bundle: dict, d: pd.DataFrame) -> np.ndarray:
    x = bundle['imputer'].transform(log_x(d, bundle['features']))
    if bundle['kind'] == 'MLP_64x32':
        x = bundle['scaler'].transform(x)
        return np.mean([m.predict(x) for m in bundle['models']], axis=0)
    return bundle['model'].predict(x)


def fit_model(name: str, train: pd.DataFrame, features: list[str], seed: int,
              fast: bool, selection: bool = False, initializations: int = 1,
              final: bool = False):
    if name == 'MLP_64x32':
        max_epochs = 45 if fast else (130 if selection else 420)
        patience = 7 if fast else (18 if selection else 55)
        return fit_mlp_group_early_stop(
            train, features, seed, fast, max_epochs=max_epochs, patience=patience,
            initializations=initializations, alpha=0.20
        )
    return fit_extratrees(train, features, seed, fast, final=final)


def evaluate_feature_set(d: pd.DataFrame, features: list[str], args, seed: int,
                         deadline: float, cache: dict) -> dict | None:
    key = tuple(features)
    if key in cache:
        return cache[key]
    rows = []
    repeats = 1 if args.fast else args.selection_repeats
    folds = 3 if args.fast else args.selection_folds
    for repeat in range(1, repeats + 1):
        for fold, held, tr, te in site_folds(d, folds, seed + repeat * 10007):
            if time.monotonic() >= deadline:
                return None
            for model_name in ('MLP_64x32', 'ExtraTrees'):
                bundle, info = fit_model(
                    model_name, tr, features,
                    seed + repeat * 100000 + fold * 1000 + (0 if model_name.startswith('MLP') else 500),
                    args.fast, selection=True, initializations=1
                )
                pred = predict_bundle(bundle, te)
                g = aggregate_groups(te, pred)
                m = metrics_from_groups(g)
                rows.append({'模型': model_name, '轮次': repeat, '折': fold,
                             '测试地点': '；'.join(held), **m})
    rdf = pd.DataFrame(rows)
    mlp = rdf[rdf['模型'] == 'MLP_64x32']
    et = rdf[rdf['模型'] == 'ExtraTrees']
    result = {
        '特征': list(features), '特征数': len(features),
        'MLP_MSE': float(mlp.MSE.mean()), 'MLP_MSE标准差': float(mlp.MSE.std(ddof=1)),
        'ExtraTrees_MSE': float(et.MSE.mean()), 'ExtraTrees_MSE标准差': float(et.MSE.std(ddof=1)),
        '综合MSE': float(0.75 * mlp.MSE.mean() + 0.25 * et.MSE.mean()),
        '完成折数': int(len(mlp)), '逐折': rows,
    }
    cache[key] = result
    return result


def removal_priority(d: pd.DataFrame, features: list[str], seed: int) -> list[str]:
    # 仅用于决定优先试删顺序；真正是否删除仍由整地点交叉验证决定。
    gx = np.log10(d[features].astype(float)).groupby(d[GROUP]).median()
    gx = gx.fillna(gx.median()).fillna(0.0)
    gy = d.groupby(GROUP)[TARGET].first().reindex(gx.index).to_numpy(float)
    model = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=2,
                                max_features=0.8, random_state=seed, n_jobs=-1)
    model.fit(gx.to_numpy(), gy)
    importance = dict(zip(features, model.feature_importances_))
    corr = gx.corr(method='spearman').abs().fillna(0)
    redundancy = {f: float(corr.loc[f].drop(f).max()) if len(features) > 1 else 0 for f in features}
    # 低重要性且与其他元素高度冗余者优先尝试删除。
    return sorted(features, key=lambda f: (importance[f] - 0.08 * redundancy[f], f))


def backward_elimination(dev: pd.DataFrame, args, out: Path, deadline: float):
    current = list(FEATURES)
    cache: dict = {}
    path_rows: list[dict] = []
    candidate_rows: list[dict] = []
    baseline = evaluate_feature_set(dev, current, args, SEED + 501, deadline, cache)
    if baseline is None:
        raise RuntimeError('时间不足以完成17元素基线筛选，请增加训练时间。')
    path_rows.append({'轮次': 0, '删除元素': '', **{k: v for k, v in baseline.items() if k not in ('逐折', '特征')},
                      '保留元素': '、'.join(current), '接受删除': True})
    current_score = baseline
    max_rounds = 1 if args.fast else args.max_elimination_rounds
    for round_no in range(1, max_rounds + 1):
        if len(current) <= args.min_features or time.monotonic() >= deadline:
            break
        order = removal_priority(dev, current, SEED + round_no * 1237)
        pool = order[:min(len(order), 3 if args.fast else args.candidate_pool)]
        tested = []
        for removed in pool:
            if time.monotonic() >= deadline:
                break
            candidate = [f for f in current if f != removed]
            score = evaluate_feature_set(dev, candidate, args, SEED + 501, deadline, cache)
            if score is None:
                break
            row = {'轮次': round_no, '尝试删除': removed, '尝试删除符号': SYMBOL[removed],
                   '候选保留元素': '、'.join(candidate),
                   **{k: v for k, v in score.items() if k not in ('逐折', '特征')}}
            candidate_rows.append(row)
            tested.append((score['综合MSE'], score['MLP_MSE'], removed, candidate, score))
            pd.DataFrame(candidate_rows).to_csv(out / '逐元素消融候选结果.csv', index=False, encoding='utf-8-sig')
        if not tested:
            break
        tested.sort(key=lambda x: (x[0], x[1], x[2]))
        _, _, removed, candidate, best = tested[0]
        improvement = current_score['综合MSE'] - best['综合MSE']
        mlp_change = best['MLP_MSE'] - current_score['MLP_MSE']
        accept = improvement >= args.min_improvement and mlp_change <= args.max_mlp_worsening
        path_rows.append({
            '轮次': round_no, '删除元素': removed, '删除元素符号': SYMBOL[removed],
            **{k: v for k, v in best.items() if k not in ('逐折', '特征')},
            '综合MSE改善': improvement, 'MLP_MSE变化': mlp_change,
            '保留元素': '、'.join(candidate if accept else current), '接受删除': accept,
        })
        pd.DataFrame(path_rows).to_csv(out / '逐元素消融路径.csv', index=False, encoding='utf-8-sig')
        if not accept:
            break
        current = candidate
        current_score = best
    return current, pd.DataFrame(path_rows), pd.DataFrame(candidate_rows), cache


def run_repeated_cv(data: pd.DataFrame, selected: list[str], args, out: Path,
                    deadline: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_sets = [('筛选后元素', selected)]
    if selected != FEATURES:
        feature_sets.append(('17元素基线', list(FEATURES)))
    all_rows: list[dict] = []
    all_groups: list[pd.DataFrame] = []
    last_repeat_seconds = None
    max_repeats = 1 if args.fast else args.max_final_repeats
    for repeat in range(1, max_repeats + 1):
        remain = deadline - time.monotonic()
        if repeat > 1 and last_repeat_seconds and remain < last_repeat_seconds * 1.15:
            break
        repeat_start = time.monotonic()
        temp_rows, temp_groups = [], []
        complete = True
        for fold, held, tr, te in site_folds(data, args.folds, SEED + 900000 + repeat * 10007):
            for set_name, features in feature_sets:
                for model_name in ('MLP_64x32', 'ExtraTrees'):
                    if time.monotonic() >= deadline:
                        complete = False
                        break
                    seed = SEED + repeat * 1000000 + fold * 10000 + len(features) * 100 + (0 if model_name.startswith('MLP') else 50)
                    bundle, info = fit_model(
                        model_name, tr, features, seed, args.fast, selection=False,
                        initializations=(1 if args.fast else args.initializations)
                    )
                    pred = predict_bundle(bundle, te)
                    train_pred = predict_bundle(bundle, tr)
                    g = aggregate_groups(te, pred)
                    gt = aggregate_groups(tr, train_pred)
                    m = metrics_from_groups(g)
                    mt = metrics_from_groups(gt)
                    temp_rows.append({
                        '轮次': repeat, '折': fold, '元素方案': set_name, '模型': model_name,
                        '特征数': len(features), '实际元素': '、'.join(features),
                        '测试地点': '；'.join(held), '测试组数': len(g),
                        'R2': m['R2'], 'MSE': m['MSE'], 'RMSE': m['RMSE'],
                        '训练R2': mt['R2'], '训练MSE': mt['MSE'], '训练RMSE': mt['RMSE'],
                        'MLP早停轮数': json.dumps(bundle.get('epochs', []), ensure_ascii=False),
                    })
                    g.insert(0, '轮次', repeat); g.insert(1, '折', fold)
                    g.insert(2, '元素方案', set_name); g.insert(3, '模型', model_name)
                    temp_groups.append(g)
                if not complete:
                    break
            if not complete:
                break
        if not complete:
            break
        all_rows.extend(temp_rows); all_groups.extend(temp_groups)
        pd.DataFrame(all_rows).to_csv(out / '整地点逐折结果.csv', index=False, encoding='utf-8-sig')
        pd.concat(all_groups, ignore_index=True).to_csv(out / '整地点逐组预测.csv', index=False, encoding='utf-8-sig')
        last_repeat_seconds = time.monotonic() - repeat_start
        print(f'完成正式整地点验证第{repeat}轮，用时{last_repeat_seconds/60:.1f}分钟。', flush=True)
    if not all_rows:
        raise RuntimeError('没有完成一轮正式整地点验证，请增加时间预算或使用快速检查。')
    return pd.DataFrame(all_rows), pd.concat(all_groups, ignore_index=True)


def summarize_cv(rows: pd.DataFrame, groups: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_summary = rows.groupby(['元素方案', '模型'], as_index=False).agg(
        平均折R2=('R2', 'mean'), R2标准差=('R2', 'std'),
        平均折MSE=('MSE', 'mean'), MSE标准差=('MSE', 'std'),
        平均折RMSE=('RMSE', 'mean'), 平均训练MSE=('训练MSE', 'mean'),
        完成折数=('折', 'size'), 完成轮数=('轮次', 'nunique')
    )
    agg = groups.groupby(['元素方案', '模型', GROUP, SITE], as_index=False).agg(
        真实值=('真实值', 'first'), 预测值=('预测值', 'mean'), 预测标准差=('预测值', 'std'),
        预测次数=('预测值', 'size')
    )
    summary_rows = []
    for (set_name, model), sub in agg.groupby(['元素方案', '模型']):
        m = metrics_from_groups(sub)
        summary_rows.append({'元素方案': set_name, '模型': model,
                             '按组汇总R2': m['R2'], '按组汇总MSE': m['MSE'],
                             '按组汇总RMSE': m['RMSE'], '汇总组数': len(sub)})
    summary = fold_summary.merge(pd.DataFrame(summary_rows), on=['元素方案', '模型'], how='left')
    return summary, agg


def lockbox_test(dev: pd.DataFrame, lockbox: pd.DataFrame, selected: list[str], args, out: Path):
    rows, groups = [], []
    for model_name in ('MLP_64x32', 'ExtraTrees'):
        bundle, info = fit_model(
            model_name, dev, selected, SEED + 700000 + (0 if model_name.startswith('MLP') else 100),
            args.fast, selection=False, initializations=(1 if args.fast else args.initializations)
        )
        pred = predict_bundle(bundle, lockbox)
        g = aggregate_groups(lockbox, pred)
        m = metrics_from_groups(g)
        rows.append({'模型': model_name, 'R2': m['R2'], 'MSE': m['MSE'], 'RMSE': m['RMSE'],
                     '锁定地点数': lockbox[SITE].nunique(), '标签组数': len(g),
                     '锁定地点': '；'.join(sorted(lockbox[SITE].unique()))})
        g.insert(0, '模型', model_name)
        groups.append(g)
    rdf = pd.DataFrame(rows); gdf = pd.concat(groups, ignore_index=True)
    rdf.to_csv(out / '锁定地点独立测试指标.csv', index=False, encoding='utf-8-sig')
    gdf.to_csv(out / '锁定地点逐组预测.csv', index=False, encoding='utf-8-sig')
    return rdf, gdf


def find_column(columns, aliases):
    normalized = {str(c).strip().lower().replace(' ', '').replace('_', ''): c for c in columns}
    for alias in aliases:
        key = str(alias).strip().lower().replace(' ', '').replace('_', '')
        if key in normalized:
            return normalized[key]
    return None


def load_external(path: Path) -> tuple[pd.DataFrame, str | None, str]:
    raw = read_table(path)
    out = pd.DataFrame(index=raw.index)
    for feature in FEATURES:
        col = find_column(raw.columns, FEATURE_ALIASES[feature])
        if col is not None:
            out[feature] = pd.to_numeric(raw[col], errors='coerce')
        else:
            out[feature] = np.nan
    out[FEATURES] = out[FEATURES].mask(out[FEATURES] <= 0)
    target_col = find_column(raw.columns, TARGET_ALIASES)
    if target_col is not None:
        out[TARGET] = pd.to_numeric(raw[target_col], errors='coerce')
    id_col = find_column(raw.columns, ID_ALIASES)
    out['外部样品编号'] = raw[id_col].astype(str) if id_col is not None else [f'JIM_{i+1:04d}' for i in range(len(raw))]
    group_col = find_column(raw.columns, EXTERNAL_GROUP_ALIASES)
    out['外部分组'] = raw[group_col].astype(str) if group_col is not None else out['外部样品编号']
    if SITE in raw.columns:
        out[SITE] = raw[SITE].astype(str)
    else:
        out[SITE] = 'JIM外部数据'
    out[GROUP] = out['外部分组']
    out[SD] = np.nan
    return out.reset_index(drop=True), target_col, str(id_col or '自动编号')


def find_external_file(explicit: str) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f'外部验证文件不存在：{p}')
        return p
    candidates = []
    for pattern in ('JIM*.csv', 'JIM*.xlsx', 'jim*.csv', 'jim*.xlsx'):
        candidates.extend(ROOT.glob(pattern))
    candidates = [p for p in candidates if '模板' not in p.name and 'template' not in p.name.lower()]
    return sorted(candidates)[0] if candidates else None


def external_validation(path: Path | None, models: dict, train: pd.DataFrame,
                        selected: list[str], out: Path):
    if path is None:
        (out / 'JIM外部验证_未运行.txt').write_text(
            '未检测到JIM数据。请将JIM CSV/XLSX放到训练包根目录，文件名以JIM开头，再重新运行；'
            '或用 --external 指定路径。JIM不会参与元素筛选、调参或训练。\n', encoding='utf-8'
        )
        return None, None
    ext, target_col, id_col = load_external(path)
    missing_all = [f for f in selected if ext[f].isna().all()]
    if missing_all:
        raise ValueError('JIM数据中以下最终元素整列缺失：' + '、'.join(missing_all))
    train_log = pd.DataFrame(log_x(train, selected), columns=selected)
    low, high = train_log.quantile(0.01), train_log.quantile(0.99)
    ext_log = pd.DataFrame(log_x(ext, selected), columns=selected)
    outside = ((ext_log.lt(low)) | (ext_log.gt(high))).sum(axis=1)
    pred_table = ext[['外部样品编号', '外部分组']].copy()
    pred_table['适用域外元素数'] = outside.to_numpy(int)
    pred_table['适用域提示'] = np.where(outside >= max(2, math.ceil(len(selected) * 0.2)), '谨慎解释', '域内或接近域内')
    if TARGET in ext.columns:
        pred_table['真实ΔFMQ'] = ext[TARGET]
    metric_rows = []
    for name, bundle in models.items():
        pred = predict_bundle(bundle, ext)
        pred_table[f'{name}_预测ΔFMQ'] = pred
        if TARGET in ext.columns and ext[TARGET].notna().sum() >= 1:
            valid = ext[TARGET].notna()
            g = aggregate_groups(ext.loc[valid].reset_index(drop=True), pred[valid.to_numpy()])
            m = metrics_from_groups(g)
            metric_rows.append({'模型': name, 'R2': m['R2'], 'MSE': m['MSE'], 'RMSE': m['RMSE'],
                                '外部标签组数': len(g), '外部样品数': int(valid.sum()),
                                '外部数据文件': path.name, '目标列': str(target_col), '编号列': id_col})
    pred_table.to_csv(out / 'JIM外部验证逐样品预测.csv', index=False, encoding='utf-8-sig')
    metrics = pd.DataFrame(metric_rows)
    if not metrics.empty:
        metrics.to_csv(out / 'JIM独立外部验证指标.csv', index=False, encoding='utf-8-sig')
    (out / 'JIM外部验证声明.txt').write_text(
        'JIM未参与训练、元素筛选、超参数选择或停止条件；仅在模型锁定后预测一次。'
        '若看到本次JIM结果后继续据此调参，后续应将其称为外部对比集而非盲法独立验证集。\n',
        encoding='utf-8'
    )
    return metrics, pred_table


def feature_importance_table(bundle: dict) -> pd.DataFrame:
    model = bundle['model']
    n_original = len(bundle['features'])
    importance = model.feature_importances_[:n_original]
    out = pd.DataFrame({'元素': bundle['features'], '元素符号': [SYMBOL[x] for x in bundle['features']],
                        'ExtraTrees重要性': importance})
    return out.sort_values('ExtraTrees重要性', ascending=False).reset_index(drop=True)


def plot_outputs(out: Path, selected: list[str], chosen_set_name: str, path_df: pd.DataFrame,
                 candidate_df: pd.DataFrame, summary: pd.DataFrame,
                 agg: pd.DataFrame, lock_metrics: pd.DataFrame,
                 lock_groups: pd.DataFrame, importance: pd.DataFrame,
                 external_metrics: pd.DataFrame | None, external_pred: pd.DataFrame | None):
    set_plot_style()
    figdir = out / '图片'
    figdir.mkdir(exist_ok=True)

    if not path_df.empty:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        x = path_df['特征数'].astype(int)
        ax.plot(x, path_df['MLP_MSE'], 'o-', label='MLP 64→32', lw=2)
        ax.plot(x, path_df['ExtraTrees_MSE'], 's-', label='ExtraTrees', lw=2)
        ax.axhline(0.4, color='#b22222', ls='--', label='目标 MSE=0.4')
        ax.invert_xaxis(); ax.set_xlabel('保留元素数'); ax.set_ylabel('筛选阶段整地点 MSE')
        ax.set_title('逐元素向后消融路径'); ax.grid(alpha=.25); ax.legend()
        fig.tight_layout(); fig.savefig(figdir / '01_逐元素消融路径.png', dpi=220); plt.close(fig)

    if not candidate_df.empty:
        last_round = int(candidate_df['轮次'].max())
        sub = candidate_df[candidate_df['轮次'] == last_round].sort_values('MLP_MSE')
        fig, ax = plt.subplots(figsize=(9, max(4.5, len(sub) * .48)))
        ax.barh(sub['尝试删除符号'], sub['MLP_MSE'], color='#3b82b8')
        ax.axvline(0.4, color='#b22222', ls='--')
        ax.invert_yaxis(); ax.set_xlabel('删除该元素后的 MLP 整地点 MSE')
        ax.set_title(f'第{last_round}轮消融候选比较'); ax.grid(axis='x', alpha=.25)
        fig.tight_layout(); fig.savefig(figdir / '02_最后一轮消融候选.png', dpi=220); plt.close(fig)

    selected_summary = summary[summary['元素方案'] == chosen_set_name].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, col, title in zip(axes, ['按组汇总R2', '平均折MSE', '按组汇总RMSE'], ['R²', 'MSE', 'RMSE']):
        bars = ax.bar(selected_summary['模型'], selected_summary[col], color=['#ef7f1a', '#3b82b8'])
        ax.set_title(f'最终元素方案：{title}'); ax.grid(axis='y', alpha=.25)
        if col == '平均折MSE': ax.axhline(.4, color='#b22222', ls='--')
        for b, v in zip(bars, selected_summary[col]):
            ax.text(b.get_x()+b.get_width()/2, b.get_height(), f'{v:.3f}', ha='center', va='bottom')
    fig.tight_layout(); fig.savefig(figdir / '03_双模型正式指标.png', dpi=220); plt.close(fig)

    for model_name in ('MLP_64x32', 'ExtraTrees'):
        sub = agg[(agg['元素方案'] == chosen_set_name) & (agg['模型'] == model_name)].copy()
        if sub.empty: continue
        fig, ax = plt.subplots(figsize=(7, 6.3))
        ax.errorbar(sub['真实值'], sub['预测值'], yerr=sub['预测标准差'].fillna(0), fmt='o',
                    color='#ef7f1a', ecolor='#b8b8b8', alpha=.85, capsize=2)
        lo = min(sub['真实值'].min(), sub['预测值'].min()) - .2
        hi = max(sub['真实值'].max(), sub['预测值'].max()) + .2
        ax.plot([lo, hi], [lo, hi], '--', color='#b22222', label='理想预测线')
        m = metrics_from_groups(sub)
        ax.text(.04, .96, f"R²={m['R2']:.3f}\nMSE={m['MSE']:.3f}\nRMSE={m['RMSE']:.3f}",
                transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', fc='white', alpha=.85))
        ax.set(xlabel='真实 ΔFMQ', ylabel='整地点预测 ΔFMQ', title=f'{model_name}：真实值与整地点预测值')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.grid(alpha=.25); ax.legend()
        fig.tight_layout(); fig.savefig(figdir / f'04_{model_name}_真实值与整地点预测值.png', dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(selected_summary)); width = .34
    ax.bar(x-width/2, selected_summary['平均训练MSE'], width, label='训练地点MSE', color='#9ecae1')
    ax.bar(x+width/2, selected_summary['平均折MSE'], width, label='未见地点MSE', color='#3182bd')
    ax.set_xticks(x, selected_summary['模型']); ax.set_ylabel('MSE'); ax.set_title('训练与未见地点误差：过拟合诊断')
    ax.grid(axis='y', alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(figdir / '05_训练与测试MSE_过拟合诊断.png', dpi=220); plt.close(fig)

    err = agg[agg['元素方案'] == chosen_set_name].groupby(['模型', SITE], as_index=False).agg(
        地点平均绝对误差=('预测值', lambda s: np.nan),
    )
    # 单独重算，避免 groupby lambda 丢失真实值。
    tmp = agg[agg['元素方案'] == chosen_set_name].copy()
    tmp['绝对误差'] = (tmp['预测值'] - tmp['真实值']).abs()
    err = tmp.groupby(['模型', SITE], as_index=False)['绝对误差'].mean()
    pivot = err.pivot(index=SITE, columns='模型', values='绝对误差').fillna(0)
    pivot['最大误差'] = pivot.max(axis=1); pivot = pivot.sort_values('最大误差').tail(12).drop(columns='最大误差')
    pivot.plot(kind='barh', figsize=(10, 6), color=['#3b82b8', '#ef7f1a'])
    plt.xlabel('地点平均绝对误差 |ΔFMQ|'); plt.ylabel('地点'); plt.title('最难预测的12个地点')
    plt.grid(axis='x', alpha=.25); plt.tight_layout(); plt.savefig(figdir / '06_困难地点.png', dpi=220); plt.close()

    imp = importance.sort_values('ExtraTrees重要性')
    fig, ax = plt.subplots(figsize=(8, max(5, len(imp) * .38)))
    ax.barh(imp['元素符号'], imp['ExtraTrees重要性'], color='#3b82b8')
    ax.set_xlabel('ExtraTrees重要性'); ax.set_title('最终保留元素的重要性')
    ax.grid(axis='x', alpha=.25); fig.tight_layout(); fig.savefig(figdir / '07_最终元素重要性.png', dpi=220); plt.close(fig)

    if not lock_metrics.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        bars = ax.bar(lock_metrics['模型'], lock_metrics['MSE'], color=['#ef7f1a', '#3b82b8'])
        ax.axhline(.4, color='#b22222', ls='--', label='目标 MSE=0.4')
        for b, v in zip(bars, lock_metrics['MSE']):
            ax.text(b.get_x()+b.get_width()/2, v, f'{v:.3f}', ha='center', va='bottom')
        ax.set_ylabel('MSE'); ax.set_title('未参与元素筛选的锁定地点测试')
        ax.grid(axis='y', alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(figdir / '08_锁定地点独立测试.png', dpi=220); plt.close(fig)

    if external_metrics is not None and not external_metrics.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        axes[0].bar(external_metrics['模型'], external_metrics['MSE'], color=['#ef7f1a', '#3b82b8'])
        axes[0].set_title('JIM独立外部验证 MSE'); axes[0].grid(axis='y', alpha=.25)
        axes[1].bar(external_metrics['模型'], external_metrics['RMSE'], color=['#ef7f1a', '#3b82b8'])
        axes[1].set_title('JIM独立外部验证 RMSE'); axes[1].grid(axis='y', alpha=.25)
        fig.tight_layout(); fig.savefig(figdir / '09_JIM独立外部验证.png', dpi=220); plt.close(fig)


def write_summary(out: Path, selected: list[str], path_df: pd.DataFrame,
                  cv_summary: pd.DataFrame, lock_metrics: pd.DataFrame,
                  external_metrics: pd.DataFrame | None, elapsed: float,
                  lock_sites: list[str], decision_note: str, args):
    lines = [
        '# 元素筛选、整地点训练与外部验证结果', '',
        f'- 总用时：{elapsed/60:.1f}分钟；设定时间预算：{args.minutes:.0f}分钟。',
        f'- 最终保留{len(selected)}个元素：' + '、'.join(selected),
        f'- 元素定型判定：{decision_note}',
        f'- 锁定地点（完全不参与元素筛选）：' + '；'.join(lock_sites),
        '- 正式模型：MLP 64→32与ExtraTrees；未运行随机颗粒划分。',
        '- 正式指标：R²、MSE、RMSE。', '',
        '## 整地点重复交叉验证', '',
        '```text', cv_summary.to_string(index=False), '```', '',
        '## 锁定地点独立测试', '',
        '```text', lock_metrics.to_string(index=False), '```', '',
    ]
    if external_metrics is not None and not external_metrics.empty:
        lines.extend(['## JIM独立外部验证', '', '```text', external_metrics.to_string(index=False), '```', '',
                      'JIM未参与训练、元素筛选和超参数选择。'])
    else:
        lines.extend(['## JIM独立外部验证', '', '本次未检测到JIM数据，因此未运行。把JIM表放入训练包后可自动执行。'])
    (out / '先看这里_结果总结.md').write_text('\n'.join(lines), encoding='utf-8')


def run(args):
    start = time.monotonic()
    input_path = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw, data, excluded = load_main(input_path)
    excluded.to_csv(out / '剔除记录_共34颗.csv', index=False, encoding='utf-8-sig')
    audit = excluded.groupby([GROUP, SITE, TARGET, '剔除原因'], as_index=False).agg(剔除锆石数=(TARGET, 'size'))
    audit.to_csv(out / '剔除两组审计.csv', index=False, encoding='utf-8-sig')

    lock_sites = choose_lockbox_sites(data, args.lockbox_fraction, SEED + 2026)
    dev = data.loc[~data[SITE].isin(lock_sites)].copy().reset_index(drop=True)
    lockbox = data.loc[data[SITE].isin(lock_sites)].copy().reset_index(drop=True)
    pd.DataFrame({'锁定地点': lock_sites}).to_csv(out / '锁定地点清单.csv', index=False, encoding='utf-8-sig')

    total_seconds = args.minutes * 60
    selection_deadline = start + total_seconds * (0.35 if args.fast else args.selection_time_fraction)
    final_cv_deadline = start + total_seconds * (0.78 if args.fast else 0.86)
    config = vars(args).copy()
    config.update({'训练数据SHA256': sha256(input_path), '原始记录数': len(raw), '入模记录数': len(data),
                   '标签组数': data[GROUP].nunique(), '地点数': data[SITE].nunique(),
                   '开发地点数': dev[SITE].nunique(), '锁定地点数': lockbox[SITE].nunique(),
                   '锁定地点': lock_sites, '随机划分': False})
    (out / '运行配置.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')

    print('阶段1/5：在开发地点内进行逐元素向后消融。', flush=True)
    selected, path_df, candidate_df, cache = backward_elimination(dev, args, out, selection_deadline)
    (out / '筛选阶段候选元素.txt').write_text('\n'.join(selected) + '\n', encoding='utf-8')

    print('阶段2/5：使用固定整地点折进行双模型重复验证。', flush=True)
    cv_rows, cv_groups = run_repeated_cv(data, selected, args, out, final_cv_deadline)
    cv_summary, agg = summarize_cv(cv_rows, cv_groups)
    cv_summary.to_csv(out / '模型指标汇总.csv', index=False, encoding='utf-8-sig')
    agg.to_csv(out / '各组平均预测.csv', index=False, encoding='utf-8-sig')

    final_features = list(selected)
    chosen_set_name = '筛选后元素'
    decision_note = '筛选后元素在正式重复整地点验证中通过性能保护，作为最终模型输入。'
    if selected != FEATURES:
        mlp_rows = cv_summary[cv_summary['模型'] == 'MLP_64x32'].set_index('元素方案')
        selected_mse = float(mlp_rows.loc['筛选后元素', '平均折MSE'])
        baseline_mse = float(mlp_rows.loc['17元素基线', '平均折MSE'])
        if selected_mse > baseline_mse + args.final_guard_tolerance:
            final_features = list(FEATURES)
            chosen_set_name = '17元素基线'
            decision_note = (
                f'筛选候选的MLP平均折MSE={selected_mse:.4f}，17元素基线={baseline_mse:.4f}；'
                f'恶化超过保护阈值{args.final_guard_tolerance:.4f}，因此最终恢复17元素。'
            )
    (out / '最终保留元素.txt').write_text('\n'.join(final_features) + '\n', encoding='utf-8')
    (out / '元素定型判定.txt').write_text(decision_note + '\n', encoding='utf-8')

    print('阶段3/5：评估从未参与元素筛选的锁定地点。', flush=True)
    lock_metrics, lock_groups = lockbox_test(dev, lockbox, final_features, args, out)

    print('阶段4/5：使用全部合格训练数据锁定最终模型。', flush=True)
    final_models = {}
    mlp, mlp_info = fit_model('MLP_64x32', data, final_features, SEED + 990001, args.fast,
                              selection=False, initializations=(1 if args.fast else max(3, args.initializations)), final=True)
    et, et_info = fit_model('ExtraTrees', data, final_features, SEED + 990101, args.fast,
                            selection=False, initializations=1, final=True)
    final_models['MLP_64x32'] = mlp; final_models['ExtraTrees'] = et
    metadata = {'features': final_features, 'target': TARGET, 'group': GROUP, 'site': SITE,
                'training_sha256': sha256(input_path), 'excluded_groups': EXCLUDE_REASONS,
                'note': '完整训练数据锁定模型；正式性能见整地点与锁定地点结果。'}
    joblib.dump({'bundle': mlp, 'metadata': metadata}, out / 'MLP64x32_锁定模型.joblib')
    joblib.dump({'bundle': et, 'metadata': metadata}, out / 'ExtraTrees_锁定模型.joblib')
    importance = feature_importance_table(et)
    importance.to_csv(out / '最终元素重要性.csv', index=False, encoding='utf-8-sig')

    print('阶段5/5：模型锁定后运行JIM外部验证并生成图片。', flush=True)
    external_path = find_external_file(args.external)
    external_metrics, external_pred = external_validation(external_path, final_models, data, final_features, out)
    plot_outputs(out, final_features, chosen_set_name, path_df, candidate_df, cv_summary, agg,
                 lock_metrics, lock_groups, importance, external_metrics, external_pred)
    elapsed = time.monotonic() - start
    write_summary(out, final_features, path_df, cv_summary, lock_metrics,
                  external_metrics, elapsed, lock_sites, decision_note, args)
    print(f'全部完成，用时{elapsed/60:.1f}分钟。结果：{out}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='逐元素消融 + 双模型整地点训练 + JIM外部验证')
    ap.add_argument('--input', default=str(ROOT / '入模数据_宽松筛选.csv'))
    ap.add_argument('--external', default='', help='JIM CSV/XLSX；留空时自动寻找同目录下以JIM开头的文件')
    ap.add_argument('--output', default=str(ROOT / '元素筛选_双模型_整地点与JIM结果'))
    ap.add_argument('--minutes', type=float, default=110, help='总训练预算，建议90–120分钟')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--selection-folds', type=int, default=4)
    ap.add_argument('--selection-repeats', type=int, default=2)
    ap.add_argument('--candidate-pool', type=int, default=7, help='每轮优先试删的元素数')
    ap.add_argument('--max-elimination-rounds', type=int, default=8)
    ap.add_argument('--min-features', type=int, default=6)
    ap.add_argument('--min-improvement', type=float, default=0.002)
    ap.add_argument('--max-mlp-worsening', type=float, default=0.003)
    ap.add_argument('--final-guard-tolerance', type=float, default=0.002,
                    help='正式重复整地点验证中，筛选方案相对17元素允许的最大MLP MSE恶化')
    ap.add_argument('--selection-time-fraction', type=float, default=0.48)
    ap.add_argument('--lockbox-fraction', type=float, default=0.20)
    ap.add_argument('--initializations', type=int, default=2)
    ap.add_argument('--max-final-repeats', type=int, default=12)
    ap.add_argument('--fast', action='store_true', help='约2–5分钟流程检查，不作为正式结果')
    run(ap.parse_args())
