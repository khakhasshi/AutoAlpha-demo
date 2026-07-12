from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


HORIZON: int = 20
LABEL_KIND: str = 'rank'
FACTOR_NAME: str = 'demo_v1_h20_rank_composite_icir_decay'

ITER_NOTE: dict = {
    'op_type': 'combine_method',
    'hypothesis': '将指数衰减半衰期从63缩短至21，给予近期因子IC表现更高权重，以适应快速变化的市场风格，期望小幅提升score。',
    'change': "修改 _icir_weights 函数的 decay_halflife 默认参数为21。",
    'expected': 'score提升0.01~0.04。',
    'parent_iter': 139,
    'reasoning': '当前半衰期63(~1季度)可能仍不能及时反映因子表现变化；进一步缩短至21(~1个月)可能更快捕捉IC动量，在trade_v2高换手惩罚下仍可能提升整体质量。'
}


def cs_rank_zscore(f: pd.DataFrame) -> pd.DataFrame:
    """Daily percentile rank -> inverse normal transform."""
    pct = f.rank(axis=1, pct=True, method='average').clip(1e-10, 1 - 1e-10)
    z = pd.DataFrame(norm.ppf(pct.values), index=pct.index, columns=pct.columns)
    return z


def _pivot(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    return panel.pivot(index='date', columns='symbol', values=col).sort_index()


def f_reversal_1(panel: pd.DataFrame) -> pd.DataFrame:
    close = _pivot(panel, 'close')
    return -close.pct_change(1, fill_method=None)


def f_volatility_10(panel: pd.DataFrame) -> pd.DataFrame:
    close = _pivot(panel, 'close')
    ret = close.pct_change(fill_method=None)
    return -ret.rolling(10, min_periods=5).std()


def f_amihud_20(panel: pd.DataFrame) -> pd.DataFrame:
    close = _pivot(panel, 'close')
    volume = _pivot(panel, 'volume')
    amount = close * volume
    ret_abs = close.pct_change(fill_method=None).abs()
    illiq = (ret_abs / amount.replace(0, np.nan)).rolling(20, min_periods=10).mean()
    return -illiq


def f_hl_range_10(panel: pd.DataFrame) -> pd.DataFrame:
    high = _pivot(panel, 'high')
    low = _pivot(panel, 'low')
    close = _pivot(panel, 'close')
    rng = (high - low) / close.replace(0, np.nan)
    return -rng.rolling(10, min_periods=5).mean()


def f_gap_reversal_5(panel: pd.DataFrame) -> pd.DataFrame:
    open_ = _pivot(panel, 'open')
    close = _pivot(panel, 'close')
    gap = open_ / close.shift(1).replace(0, np.nan) - 1.0
    return -gap.rolling(5, min_periods=3).mean()


def f_rsi_14(panel: pd.DataFrame) -> pd.DataFrame:
    close = _pivot(panel, 'close')
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.rolling(14, min_periods=10).mean()
    avg_down = down.rolling(14, min_periods=10).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return -rsi


def f_skew_20(panel: pd.DataFrame) -> pd.DataFrame:
    close = _pivot(panel, 'close')
    ret = close.pct_change(fill_method=None)
    skew = ret.rolling(20, min_periods=10).skew()
    return -skew


def f_volume_reversal_5(panel: pd.DataFrame) -> pd.DataFrame:
    volume = _pivot(panel, 'volume')
    mean_vol_5 = volume.rolling(5, min_periods=3).mean()
    ratio = volume / mean_vol_5.replace(0, np.nan)
    return -ratio


def f_volume_volatility_10(panel: pd.DataFrame) -> pd.DataFrame:
    volume = _pivot(panel, 'volume')
    vol_change = volume.pct_change(fill_method=None)
    return -vol_change.rolling(10, min_periods=5).std()


def f_upper_shadow_ratio_5(panel: pd.DataFrame) -> pd.DataFrame:
    high = _pivot(panel, 'high')
    low = _pivot(panel, 'low')
    open_ = _pivot(panel, 'open')
    close = _pivot(panel, 'close')
    upper_shadow = high - np.maximum(open_, close)
    total_range = high - low
    with np.errstate(invalid='ignore'):
        ratio = upper_shadow / total_range
    avg_ratio = ratio.rolling(5, min_periods=3).mean()
    return -avg_ratio


def f_lower_shadow_ratio_5(panel: pd.DataFrame) -> pd.DataFrame:
    high = _pivot(panel, 'high')
    low = _pivot(panel, 'low')
    open_ = _pivot(panel, 'open')
    close = _pivot(panel, 'close')
    lower_shadow = np.minimum(open_, close) - low
    total_range = high - low
    with np.errstate(invalid='ignore'):
        ratio = lower_shadow / total_range
    avg_ratio = ratio.rolling(5, min_periods=3).mean()
    return avg_ratio


def f_max_ret_10(panel: pd.DataFrame) -> pd.DataFrame:
    close = _pivot(panel, 'close')
    ret = close.pct_change(fill_method=None)
    max_ret = ret.rolling(10, min_periods=5).max()
    return -max_ret


def f_idio_vol_10(panel: pd.DataFrame) -> pd.DataFrame:
    '''Idiosyncratic volatility: rolling 10-day std of residual returns
    from equal-weight market model. Negated – low idio vol predicts higher returns.'''
    close = _pivot(panel, 'close')
    ret = close.pct_change(fill_method=None)
    mkt_ret = ret.mean(axis=1, skipna=True)
    residual = ret.sub(mkt_ret, axis=0)
    idio_vol = residual.rolling(10, min_periods=5).std()
    return -idio_vol


def f_price_range_position_10(panel: pd.DataFrame) -> pd.DataFrame:
    close = _pivot(panel, 'close')
    high = _pivot(panel, 'high')
    low = _pivot(panel, 'low')
    roll_high = high.rolling(10, min_periods=5).max()
    roll_low = low.rolling(10, min_periods=5).min()
    range_hl = roll_high - roll_low
    with np.errstate(invalid='ignore'):
        position = (close - roll_low) / range_hl - 0.5
    return -position


def f_momentum_20(panel: pd.DataFrame) -> pd.DataFrame:
    close = _pivot(panel, 'close')
    return close.pct_change(20, fill_method=None)


FACTORS = [
    f_reversal_1,
    f_volatility_10,
    f_amihud_20,
    f_hl_range_10,
    f_gap_reversal_5,
    f_rsi_14,
    f_skew_20,
    f_volume_reversal_5,
    f_volume_volatility_10,
    f_upper_shadow_ratio_5,
    f_lower_shadow_ratio_5,
    f_max_ret_10,
    f_idio_vol_10,
    f_price_range_position_10,
    f_momentum_20,
]


def _align_factors(factor_panels: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], pd.Index, pd.Index]:
    base = factor_panels[0]
    common_idx = base.index
    common_cols = base.columns
    for f in factor_panels[1:]:
        common_idx = common_idx.intersection(f.index)
        common_cols = common_cols.intersection(f.columns)
    aligned = [f.reindex(index=common_idx, columns=common_cols) for f in factor_panels]
    return aligned, common_idx, common_cols


def combine_equal_weight(factor_panels: list[pd.DataFrame]) -> pd.DataFrame:
    aligned, common_idx, common_cols = _align_factors(factor_panels)
    stacked = np.stack([f.values for f in aligned], axis=0)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        mean = np.nanmean(stacked, axis=0)
    return pd.DataFrame(mean, index=common_idx, columns=common_cols)


def _daily_rank_ic(signal: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    common_idx = signal.index.intersection(labels.index)
    common_cols = signal.columns.intersection(labels.columns)
    sig = signal.reindex(index=common_idx, columns=common_cols).rank(axis=1)
    lab = labels.reindex(index=common_idx, columns=common_cols).rank(axis=1)
    return sig.corrwith(lab, axis=1).dropna()


def _daily_pearson_ic(signal: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    common_idx = signal.index.intersection(labels.index)
    common_cols = signal.columns.intersection(labels.columns)
    sig = signal.reindex(index=common_idx, columns=common_cols)
    lab = labels.reindex(index=common_idx, columns=common_cols)
    return sig.corrwith(lab, axis=1).dropna()


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sv = values[order]
    sw = weights[order]
    cum = np.cumsum(sw)
    cut = 0.5 * cum[-1]
    idx = np.searchsorted(cum, cut)
    return sv[idx]


def _weighted_mad(values: np.ndarray, weights: np.ndarray, median: float) -> float:
    absdev = np.abs(values - median)
    return _weighted_median(absdev, weights)


def _icir_weights(factor_panels: list[pd.DataFrame], train_panel: pd.DataFrame, decay_halflife: int = 21) -> np.ndarray:
    import prepare
    labels = prepare.make_labels(train_panel, HORIZON, kind=LABEL_KIND)

    ic_list_rank = []
    ic_list_pearson = []
    for f in factor_panels:
        ic_r = _daily_rank_ic(f, labels)
        ic_p = _daily_pearson_ic(f, labels)
        if len(ic_r) < 20:
            ic_list_rank.append(pd.Series(dtype=float))
            ic_list_pearson.append(pd.Series(dtype=float))
        else:
            ic_list_rank.append(ic_r)
            ic_list_pearson.append(ic_p)

    # collect all non-empty dates and find latest reference date
    all_dates = pd.DatetimeIndex([])
    for s in ic_list_rank + ic_list_pearson:
        if not s.empty:
            all_dates = all_dates.union(s.index)
    if len(all_dates) == 0:
        return np.ones(len(factor_panels)) / len(factor_panels)
    ref_date = all_dates.max()
    date_diffs = (ref_date - all_dates).days
    decay_weights = np.exp(-np.log(2) * date_diffs / decay_halflife)
    decay_weights = pd.Series(decay_weights, index=all_dates)

    ir_rank = []
    ir_pearson = []
    for ic_r, ic_p in zip(ic_list_rank, ic_list_pearson):
        # rank IC
        if ic_r.empty or len(ic_r) < 20:
            ir_rank.append(0.0)
        else:
            common = ic_r.index.intersection(all_dates)
            if len(common) < 20:
                ir_rank.append(0.0)
            else:
                w = decay_weights.loc[common].values
                ic_vals = ic_r.loc[common].values
                med = _weighted_median(ic_vals, w)
                mad = _weighted_mad(ic_vals, w, med)
                sigma = mad * 1.4826
                ir = med / (sigma + 1e-8)
                ir_rank.append(ir)
        # pearson IC
        if ic_p.empty or len(ic_p) < 20:
            ir_pearson.append(0.0)
        else:
            common = ic_p.index.intersection(all_dates)
            if len(common) < 20:
                ir_pearson.append(0.0)
            else:
                w = decay_weights.loc[common].values
                ic_vals = ic_p.loc[common].values
                med = _weighted_median(ic_vals, w)
                mad = _weighted_mad(ic_vals, w, med)
                sigma = mad * 1.4826
                ir = med / (sigma + 1e-8)
                ir_pearson.append(ir)

    composite_ir = np.array(ir_rank) + 0.5 * np.array(ir_pearson)
    positive_ir = np.maximum(composite_ir, 0)
    if positive_ir.sum() == 0:
        return np.ones(len(factor_panels)) / len(factor_panels)
    weights = positive_ir / positive_ir.sum()
    return weights


def combine_icir_weight(factor_panels: list[pd.DataFrame], weights: np.ndarray) -> pd.DataFrame:
    aligned, common_idx, common_cols = _align_factors(factor_panels)
    stacked = np.stack([f.values for f in aligned], axis=0)
    w = weights.reshape(-1, 1, 1)
    valid = np.isfinite(stacked)
    weighted = np.nansum(np.where(valid, stacked, 0.0) * w, axis=0)
    scale = np.sum(valid * np.abs(w), axis=0)
    out = np.divide(weighted, scale, out=np.full_like(weighted, np.nan), where=scale > 0.0)
    return pd.DataFrame(out, index=common_idx, columns=common_cols)


def _factor_panels(panel: pd.DataFrame) -> list[pd.DataFrame]:
    out = []
    for fn in FACTORS:
        f = fn(panel)
        f = cs_rank_zscore(f)
        out.append(f)
    return out


def _finalize(signal: pd.DataFrame) -> pd.DataFrame:
    return cs_rank_zscore(signal)


def run(train_panel: pd.DataFrame, val_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fps_train = _factor_panels(train_panel)
    fps_val = _factor_panels(val_panel)
    weights = _icir_weights(fps_train, train_panel)
    sig_train_raw = combine_icir_weight(fps_train, weights)
    sig_val_raw = combine_icir_weight(fps_val, weights)
    return _finalize(sig_train_raw), _finalize(sig_val_raw)


if __name__ == '__main__':
    import prepare
    tr = prepare.load_train_panel()
    va = prepare.load_val_panel()
    print(f'[alpha] train shape = {tr.shape}, val shape = {va.shape}')
    print(f'[alpha] 因子数 = {len(FACTORS)}, HORIZON = {HORIZON}, LABEL_KIND = {LABEL_KIND}')
    sig_tr, sig_va = run(tr, va)
    print(f'[alpha] signal train shape = {sig_tr.shape}')
    print(f'[alpha] signal val   shape = {sig_va.shape}')
    print(f'[alpha] val daily mean = {sig_va.mean(axis=1).mean():+.4f}  (应满足 |mean|<0.05)')
    print(f'[alpha] val daily std  = {sig_va.std(axis=1).mean():.4f}  (应满足 0.5<std<2.0)')
    print('[alpha] OK')
