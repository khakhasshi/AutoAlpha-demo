from __future__ import annotations

import numpy as np
import pandas as pd


HORIZON: int = 1
LABEL_KIND: str = 'rank'
FACTOR_NAME: str = 'demo_v1_h1_idiovol10_13factors_no_momentum10_rev1_icir_roll126_maxret10'

ITER_NOTE: dict = {
    'op_type': 'add_factor',
    'hypothesis': 'Add f_price_range_position_10 factor to capture short-term mean-reversion from price position within 10-day high-low range. Expected to have low correlation with existing factors and improve rank_ic_ir.',
    'change': 'Added f_price_range_position_10 factor function and appended to FACTORS list. No other changes.',
    'expected': 'score improvement of +0.01-0.03 due to additional independent reversal signal.',
    'parent_iter': 100,
    'reasoning': 'Existing factors cover various reversal patterns, but a bounded range position measure (relative to recent high-low) has not been included. It captures a distinct aspect of mean-reversion with a natural 0-1 scale, likely low correlation with other oscillators.',
    'new_factor': 'f_price_range_position_10',
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


def _icir_weights(factor_panels: list[pd.DataFrame], train_panel: pd.DataFrame) -> np.ndarray:
    import prepare
    labels = prepare.make_labels(train_panel, HORIZON, kind=LABEL_KIND)

    # Compute daily rank IC for each factor
    ic_list = []
    for f in factor_panels:
        ic = _daily_rank_ic(f, labels)
        if len(ic) < 20:
            ic_list.append(pd.Series(dtype=float))
        else:
            ic_list.append(ic)

    # Align on common dates
    common_dates = None
    for ic in ic_list:
        if not ic.empty:
            if common_dates is None:
                common_dates = ic.index
            else:
                common_dates = common_dates.intersection(ic.index)

    if common_dates is None or len(common_dates) < 20:
        # fallback to equal weight
        return np.ones(len(factor_panels)) / len(factor_panels)

    # Build IC matrix [T, K]
    ic_matrix = pd.DataFrame({i: ic.reindex(common_dates) for i, ic in enumerate(ic_list)})
    ic_matrix = ic_matrix.dropna(axis=1, how='all')  # drop factors with no valid IC
    if ic_matrix.shape[1] == 0:
        return np.ones(len(factor_panels)) / len(factor_panels)

    # Keep only factors with positive mean IC (threshold 1e-5)
    mu_full = ic_matrix.mean(axis=0)
    positive_mask = mu_full > 1e-5
    if not positive_mask.any():
        return np.ones(len(factor_panels)) / len(factor_panels)

    ic_pos = ic_matrix.loc[:, positive_mask]
    mu = ic_pos.mean(axis=0).values  # shape (K_pos,)
    Sigma = np.cov(ic_pos.values, rowvar=False)  # shape (K_pos, K_pos)
    lam = 1e-3
    Sigma_reg = Sigma + lam * np.eye(Sigma.shape[0])

    try:
        w_raw = np.linalg.solve(Sigma_reg, mu)
    except np.linalg.LinAlgError:
        # fallback to equal weight
        return np.ones(len(factor_panels)) / len(factor_panels)

    # Normalize to sum to 1 (weights can be negative, but we expect positive due to positive mu)
    w = w_raw / np.sum(w_raw)

    # Map back to full factor list, zero weight for excluded factors
    full_weights = np.zeros(len(factor_panels))
    pos_indices = np.where(positive_mask)[0]
    full_weights[pos_indices] = w

    return full_weights


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
