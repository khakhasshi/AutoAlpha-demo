from __future__ import annotations

import numpy as np
import pandas as pd


HORIZON: int = 1
LABEL_KIND: str = 'rank'
FACTOR_NAME: str = 'demo_v1_h1_idiovol20_13factors_icir_roll252'

ITER_NOTE: dict = {
    'op_type': 'add_factor',
    'hypothesis': 'Add idiosyncratic volatility factor f_idio_vol_20, computed as rolling 20-day std of residual returns from market model (equal-weight market return), then negated. Low idiosyncratic volatility is known to predict higher returns, offering a signal beyond total volatility.',
    'change': 'Add f_idio_vol_20 function, compute daily stock returns and equal-weight market return, compute residual return, rolling 20-day std of residuals, negated. Append to FACTORS.',
    'expected': 'score +0.02–0.05, rank_ic_ir slightly higher. Factor expected to have low correlation with existing ones (<0.85).',
    'parent_iter': 66,
    'reasoning': 'After adding max_ret_20 gave +0.04, further factor expansion with a different economic logic (idiosyncratic risk) may continue incremental gains. Idio vol is a distinct anomaly from total vol, momentum, reversal, and shadow factors.',
    'new_factor': 'f_idio_vol_20',
}


def cs_winsorize_zscore(f: pd.DataFrame, n_mad: float = 3.0) -> pd.DataFrame:
    med = f.median(axis=1)
    mad = (f.sub(med, axis=0)).abs().median(axis=1)
    sigma = 1.4826 * mad
    lower = med - n_mad * sigma
    upper = med + n_mad * sigma
    f_w = f.clip(lower=lower, upper=upper, axis=0)
    mu = f_w.mean(axis=1)
    sd = f_w.std(axis=1).replace(0, np.nan)
    return f_w.sub(mu, axis=0).div(sd, axis=0)


def _pivot(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    return panel.pivot(index='date', columns='symbol', values=col).sort_index()


def f_reversal_3(panel: pd.DataFrame) -> pd.DataFrame:
    close = _pivot(panel, 'close')
    return -close.pct_change(3, fill_method=None)


def f_momentum_10(panel: pd.DataFrame) -> pd.DataFrame:
    close = _pivot(panel, 'close')
    return close.pct_change(10, fill_method=None)


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


def f_max_ret_20(panel: pd.DataFrame) -> pd.DataFrame:
    close = _pivot(panel, 'close')
    ret = close.pct_change(fill_method=None)
    max_ret = ret.rolling(20, min_periods=5).max()
    return -max_ret


def f_idio_vol_20(panel: pd.DataFrame) -> pd.DataFrame:
    '''Idiosyncratic volatility: rolling 20-day std of residual returns
    from equal-weight market model. Negated – low idio vol predicts higher returns.'''
    close = _pivot(panel, 'close')
    ret = close.pct_change(fill_method=None)
    # equal-weight market return (ignore NaN)
    mkt_ret = ret.mean(axis=1, skipna=True)
    # residual returns: stock return minus market return
    residual = ret.sub(mkt_ret, axis=0)
    idio_vol = residual.rolling(20, min_periods=5).std()
    return -idio_vol


FACTORS = [
    f_reversal_3,
    f_momentum_10,
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
    f_max_ret_20,
    f_idio_vol_20,  # new
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
    stacked = np.stack([f.values for f in aligned], axis=0)   # [F, T, N]
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


def _icir_weights(factor_panels: list[pd.DataFrame], train_panel: pd.DataFrame) -> np.ndarray:
    import prepare
    labels = prepare.make_labels(train_panel, HORIZON, kind=LABEL_KIND)
    raw = []
    window = 252
    for f in factor_panels:
        ic = _daily_rank_ic(f, labels)
        if len(ic) < 20:
            raw.append(0.0)
            continue
        wm = ic.rolling(window=window, min_periods=20).mean().iloc[-1]
        ws = ic.rolling(window=window, min_periods=20).std().iloc[-1]
        if np.isnan(wm) or np.isnan(ws) or ws == 0:
            raw.append(0.0)
        else:
            raw.append(wm / ws)
    w = np.asarray(raw, dtype=float)
    denom = np.nansum(np.abs(w))
    if denom <= 0.0:
        return np.ones(len(factor_panels), dtype=float) / len(factor_panels)
    return w / denom


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
        f = cs_winsorize_zscore(f)
        out.append(f)
    return out


def _finalize(signal: pd.DataFrame) -> pd.DataFrame:
    return cs_winsorize_zscore(signal)


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
