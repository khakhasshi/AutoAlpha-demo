"""
alpha.py — AutoAlpha 教学版 baseline（demo v1）
================================================================

供 AutoAlpha 讲习课演示使用。本版本是最简可运行起点：
    · 5 个经典量价因子（反转 / 动量 / 波动 / 流动性 / 振幅）
    · 截面 winsorize + zscore 标准化
    · 线性等权组合
    · 无 ML 模型训练（组合是纯线性等权，无需 fit）

学员可以由此出发做的典型练习：
    · 加一个新因子（如隔夜反转、上下影线、跳空频率）
    · 把等权改成 IC_IR 加权
    · 改 HORIZON（1 / 3 / 5 / 10 / 20）
    · 在 prepare.primary_score 里把 Sharpe / MDD 加进公式

契约（详见 program.md / evaluation.md）：
    · 必须导出 HORIZON: int ∈ prepare.ALLOWED_HORIZONS
    · 必须导出 LABEL_KIND: str ∈ prepare.ALLOWED_LABEL_KINDS
    · 必须导出 ITER_NOTE: dict（每次实验必须声明，详见 program.md §7.1）
    · 必须导出 run(train_panel, val_panel) -> (signal_train, signal_val)
    · 输出信号是 [date × symbol] 浮点 DataFrame，已做截面 winsorize + zscore
    · 严禁单股序列建模、严禁触碰 prepare 私有成员
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# =============================================================================
# 0. 顶层契约常量（runner 直接读取）
# =============================================================================
HORIZON: int = 5                                    # 持有 5 个交易日
LABEL_KIND: str = "market_neutral"                  # 扣除截面等权市场收益
FACTOR_NAME: str = "demo_v1_8factors_icir_weight"

# ITER_NOTE：每次实验必须声明（runner 强制校验）
ITER_NOTE: dict = {
    "op_type":     "add_factor",
    "hypothesis":  "新增 20 日收益偏度反转因子，捕捉极端负偏度后的短期反转，"
                   "与现有均值和标准差因子相关性低，应能增加信号多样性并提升 IC 稳定性。",
    "change":      "在 FACTORS 末尾追加 f_skew_20；其余不变。",
    "expected":    "score +0.15 ~ +0.30 左右，rank_ic_ir 与 rank_ic_mean 有望小幅提升。",
    "parent_iter": 17,
    "reasoning":   "当前 best (run 17) score 4.61，7 因子 IC_IR 加权已稳定；"
                   "偏度因子是价格分布的三阶矩，与波动率（二阶矩）和均值（一阶）正交，"
                   "加入后应提升因子的覆盖维度而不触发高相关性门控（ρ 预计 <0.4）。",
    "new_factor":  "f_skew_20",
}


# =============================================================================
# 1. 截面标准化工具（提升信噪比的关键）
# =============================================================================
def cs_winsorize_zscore(
    f: pd.DataFrame,
    n_mad: float = 3.0,
) -> pd.DataFrame:
    """
    每个截面（每一天）独立做：
        ① MAD winsorize：用 1.4826·MAD 当 σ，把超过 ±n_mad·σ 的截面外点拉回边界
        ② z-score：减截面均值，除以截面标准差

    这一步是 A 股横截面研究的信噪比基础 —— 直接把裸因子相加会被极端值主导。
    """
    med = f.median(axis=1)
    mad = (f.sub(med, axis=0)).abs().median(axis=1)
    sigma = 1.4826 * mad
    # winsorize
    lower = med - n_mad * sigma
    upper = med + n_mad * sigma
    f_w = f.clip(lower=lower, upper=upper, axis=0)
    # z-score
    mu = f_w.mean(axis=1)
    sd = f_w.std(axis=1).replace(0, np.nan)
    return f_w.sub(mu, axis=0).div(sd, axis=0)


def _pivot(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    """把长表 [date, symbol, col] 转成宽表 [date × symbol]。"""
    return panel.pivot(index="date", columns="symbol", values=col).sort_index()


# =============================================================================
# 2. 因子库 · 8 个量价因子
# =============================================================================
def f_reversal_5(panel: pd.DataFrame) -> pd.DataFrame:
    """5 日短期反转：A 股小盘股反转效应显著。
    取负号：过去涨得多的股票，未来跌回来的概率高。"""
    close = _pivot(panel, "close")
    return -close.pct_change(5, fill_method=None)


def f_momentum_20(panel: pd.DataFrame) -> pd.DataFrame:
    """20 日中期动量：中期动量效应。
    过去 20 日涨得多的股票，未来也倾向于继续涨。"""
    close = _pivot(panel, "close")
    return close.pct_change(20, fill_method=None)


def f_volatility_20(panel: pd.DataFrame) -> pd.DataFrame:
    """20 日波动率（取负）：低波动异象。
    低波动股票长期收益反而更好 —— 学界经典发现。"""
    close = _pivot(panel, "close")
    ret = close.pct_change(fill_method=None)
    return -ret.rolling(20, min_periods=10).std()


def f_amihud_20(panel: pd.DataFrame) -> pd.DataFrame:
    """20 日 Amihud 非流动性（取负）：流动性溢价。
    公式：mean(|ret| / 成交额)；取负让 “流动性好” 的因子得高分。"""
    close = _pivot(panel, "close")
    volume = _pivot(panel, "volume")
    amount = close * volume
    ret_abs = close.pct_change(fill_method=None).abs()
    illiq = (ret_abs / amount.replace(0, np.nan)).rolling(20, min_periods=10).mean()
    return -illiq


def f_hl_range_20(panel: pd.DataFrame) -> pd.DataFrame:
    """20 日日内振幅（取负）：低振幅偏好。
    (high - low) / close 的 20 日均值；取负让 “振幅小” 的因子得高分。"""
    high = _pivot(panel, "high")
    low = _pivot(panel, "low")
    close = _pivot(panel, "close")
    rng = (high - low) / close.replace(0, np.nan)
    return -rng.rolling(20, min_periods=10).mean()


def f_gap_reversal_5(panel: pd.DataFrame) -> pd.DataFrame:
    """5 日隔夜跳空反转。
    gap = open / prev_close - 1；取负号表示高开过多后短期更可能回落。"""
    open_ = _pivot(panel, "open")
    close = _pivot(panel, "close")
    gap = open_ / close.shift(1).replace(0, np.nan) - 1.0
    return -gap.rolling(5, min_periods=3).mean()


def f_rsi_14(panel: pd.DataFrame) -> pd.DataFrame:
    """14 日 RSI 反转因子（取负）。
    RSI 高（>70）=超买，未来倾向于回落；RSI 低（<30）=超卖，未来倾向于反弹。
    取负后使得预测上涨的股票（即低 RSI）因子值高。"""
    close = _pivot(panel, "close")
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.rolling(14, min_periods=10).mean()
    avg_down = down.rolling(14, min_periods=10).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return -rsi


def f_skew_20(panel: pd.DataFrame) -> pd.DataFrame:
    """20 日收益偏度反转（取负）。
    负偏度（极端下跌）之后短期倾向于反转；理论基于投资者过度反应。"""
    close = _pivot(panel, "close")
    ret = close.pct_change(fill_method=None)
    skew = ret.rolling(20, min_periods=10).skew()
    return -skew


# 因子注册表：新增因子时，在这里追加函数名即可
FACTORS = [
    f_reversal_5,
    f_momentum_20,
    f_volatility_20,
    f_amihud_20,
    f_hl_range_20,
    f_gap_reversal_5,
    f_rsi_14,
    f_skew_20,
]


# =============================================================================
# 3. 因子组合 · IC_IR 加权
# =============================================================================
def _align_factors(factor_panels: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], pd.Index, pd.Index]:
    """把多个因子面板对齐到共同的日期与股票列。"""
    base = factor_panels[0]
    common_idx = base.index
    common_cols = base.columns
    for f in factor_panels[1:]:
        common_idx = common_idx.intersection(f.index)
        common_cols = common_cols.intersection(f.columns)
    aligned = [f.reindex(index=common_idx, columns=common_cols) for f in factor_panels]
    return aligned, common_idx, common_cols


def combine_equal_weight(factor_panels: list[pd.DataFrame]) -> pd.DataFrame:
    """等权相加 —— 已假定每个因子是 zscore 后的量纲统一状态。

    实现细节：
    - 用 nanmean 而非直接相加 —— 某只股票某天可能有部分因子为 NaN，
      nanmean 会忽略 NaN 而不是把整行拉成 NaN。
    """
    aligned, common_idx, common_cols = _align_factors(factor_panels)
    stacked = np.stack([f.values for f in aligned], axis=0)   # [F, T, N]
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        mean = np.nanmean(stacked, axis=0)
    return pd.DataFrame(mean, index=common_idx, columns=common_cols)


def _daily_rank_ic(signal: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """逐日计算 rank IC。"""
    common_idx = signal.index.intersection(labels.index)
    common_cols = signal.columns.intersection(labels.columns)
    sig = signal.reindex(index=common_idx, columns=common_cols).rank(axis=1)
    lab = labels.reindex(index=common_idx, columns=common_cols).rank(axis=1)
    return sig.corrwith(lab, axis=1).dropna()


def _icir_weights(factor_panels: list[pd.DataFrame], train_panel: pd.DataFrame) -> np.ndarray:
    """用 train 段标签估计各因子的 signed IC_IR 权重。"""
    import prepare

    labels = prepare.make_labels(train_panel, HORIZON, kind=LABEL_KIND)
    raw = []
    for f in factor_panels:
        ic = _daily_rank_ic(f, labels)
        if len(ic) < 20:
            raw.append(0.0)
            continue
        sd = float(ic.std())
        raw.append(0.0 if sd == 0.0 or np.isnan(sd) else float(ic.mean() / sd))
    w = np.asarray(raw, dtype=float)
    denom = np.nansum(np.abs(w))
    if denom <= 0.0:
        return np.ones(len(factor_panels), dtype=float) / len(factor_panels)
    return w / denom


def combine_icir_weight(factor_panels: list[pd.DataFrame], weights: np.ndarray) -> pd.DataFrame:
    """按 train 段 IC_IR 权重组合因子。"""
    aligned, common_idx, common_cols = _align_factors(factor_panels)
    stacked = np.stack([f.values for f in aligned], axis=0)
    w = weights.reshape(-1, 1, 1)
    valid = np.isfinite(stacked)
    weighted = np.nansum(np.where(valid, stacked, 0.0) * w, axis=0)
    scale = np.sum(valid * np.abs(w), axis=0)
    out = np.divide(weighted, scale, out=np.full_like(weighted, np.nan), where=scale > 0.0)
    return pd.DataFrame(out, index=common_idx, columns=common_cols)


# =============================================================================
# 4. 主入口（runner 唯一调用点）
# =============================================================================
def _factor_panels(panel: pd.DataFrame) -> list[pd.DataFrame]:
    """逐因子计算 + 截面 winsorize+zscore 标准化。"""
    out = []
    for fn in FACTORS:
        f = fn(panel)
        f = cs_winsorize_zscore(f)
        out.append(f)
    return out


def _finalize(signal: pd.DataFrame) -> pd.DataFrame:
    """最终再做一次截面标准化，确保满足 prepare.validate_signal 的硬约束
    （截面 |mean|<0.05，0.5<std<2.0）。"""
    return cs_winsorize_zscore(signal)


def run(
    train_panel: pd.DataFrame,
    val_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回 (signal_train, signal_val)。

    流程（三步）：
        ① 分别算 train / val 段每个因子的截面标准化面板
        ② 用 train 段 IC_IR 权重组合因子
        ③ 再做一次截面标准化，交给 runner

    注意：权重只用 train 段标签估计；val 段只套用权重生成预测信号。
    """
    # ① 因子计算
    fps_train = _factor_panels(train_panel)
    fps_val = _factor_panels(val_panel)

    # ② IC_IR 加权组合
    weights = _icir_weights(fps_train, train_panel)
    sig_train_raw = combine_icir_weight(fps_train, weights)
    sig_val_raw = combine_icir_weight(fps_val, weights)

    # ③ 最终标准化
    return _finalize(sig_train_raw), _finalize(sig_val_raw)


# =============================================================================
# 5. CLI smoke test：python alpha.py
# =============================================================================
if __name__ == "__main__":
    import prepare
    tr = prepare.load_train_panel()
    va = prepare.load_val_panel()
    print(f"[alpha] train shape = {tr.shape}, val shape = {va.shape}")
    print(f"[alpha] 因子数 = {len(FACTORS)}, HORIZON = {HORIZON}, LABEL_KIND = {LABEL_KIND}")
    sig_tr, sig_va = run(tr, va)
    print(f"[alpha] signal train shape = {sig_tr.shape}")
    print(f"[alpha] signal val   shape = {sig_va.shape}")
    print(f"[alpha] val daily mean = {sig_va.mean(axis=1).mean():+.4f}  "
          f"(应满足 |mean|<0.05)")
    print(f"[alpha] val daily std  = {sig_va.std(axis=1).mean():.4f}  "
          f"(应满足 0.5<std<2.0)")
    print("[alpha] OK")
