# AutoAlpha — 因子构造语法（Factor Grammar）

> 目的：让 agent 生成因子时使用结构化“积木”，提升候选多样性，减少换皮重复。
> 本文件只指导 proposal 与 `alpha.py` 研究方向，不改变 `prepare.primary_score()`。

---

## 1. Proposal 必填字段

每个候选 proposal 必须声明：

```json
{
  "family": "path_quality",
  "primitive": "up_day_ratio",
  "transform": "rolling_mean_20 + cs_rank_zscore",
  "target_bottleneck": "year_stability_low"
}
```

`family` 必须优先从下表选择。3 个候选 proposal 应尽量来自 3 个不同 family。

---

## 2. 因子族

| family | 适用瓶颈 | 原语示例 | 说明 |
|---|---|---|---|
| `path_quality` | 年度稳定性、收益质量 | `up_day_ratio_N`, `trend_smoothness_N`, `drawdown_recovery_N`, `return_autocorr_N` | 关注走势路径是否平滑、是否靠少数跳涨 |
| `low_turnover_state` | 换手、年度稳定性 | `range_position_60/120/252`, `slow_vol_regime`, `amount_stability_N` | 慢变量，天然更稳定 |
| `price_volume_divergence` | 效率、低相关新信息 | `price_high_volume_fade`, `volume_confirmed_trend`, `down_volume_recovery` | 价格和成交量方向背离或确认 |
| `liquidity_shock` | 效率、回撤 | `amount_spike_N`, `volume_dryup_N`, `amihud_shock_N` | 捕捉流动性冲击后的修复或风险溢价 |
| `bar_structure` | 低相关形态 | `close_in_bar`, `upper_shadow_N`, `lower_shadow_N`, `gap_fill_ratio_N` | K 线内部结构，不要重复已有影线因子 |
| `orthogonal_residual` | 同质化、低相关 | `residualize(raw_factor, existing_signal)` | 先构造 raw idea，再保留对现有因子正交的新信息 |
| `factor_pruning` | 复杂度、同质化 | `delete_or_downweight_high_corr_factor` | 删减或降低边际贡献差的因子 |
| `signal_stability` | 年度稳定性、换手 | `ewm_signal_smoothing`, `weight_stabilization`, `top_bucket_persistence` | 组合层稳定化，不新增 raw 因子 |

---

## 3. 原语库

价格：
- `return_N`, `momentum_N`, `reversal_N`
- `range_position_N`, `high_low_position_N`
- `max_return_N`, `min_return_N`

波动：
- `volatility_N`, `downside_vol_N`, `upside_vol_N`
- `intraday_range_N`, `drawdown_N`

成交量/流动性：
- `volume_z_N`, `volume_change_N`, `amount_z_N`
- `amount_stability_N`, `amihud_N`, `liquidity_shock_N`

价量交互：
- `return_N * volume_z_N`
- `gap_N * volume_z_N`
- `intraday_return * amount_z_N`
- `range_N * volume_change_N`

路径质量：
- `up_day_ratio_N`
- `trend_smoothness_N`
- `drawdown_recovery_N`
- `return_autocorr_N`

形态：
- `close_position_in_bar`
- `upper_shadow_N`, `lower_shadow_N`
- `gap_fill_ratio_N`

---

## 4. 变换语法

截面：
- `cs_rank_zscore(x)`：默认稳健选择
- `cs_winsorize_zscore(x)`：保留强度信息时使用

时间：
- `rolling_mean(x, N)`
- `rolling_std(x, N)`
- `ewm_mean(x, span)`

非线性：
- `clip(x, lo, hi)`
- `signed_sqrt(x)`
- `log1p(abs(x)) * sign(x)`

组合：
- `x * y`
- `x / (abs(y) + eps)`
- `residualize(x, controls)`

---

## 5. 当前阶段偏好

当前 `trade_v4` best 的主要瓶颈通常来自：
- 年度稳定性偏弱；
- 收益/回撤效率仍可提升；
- 换手和复杂度暂时不应恶化。

优先生成：
- `path_quality`
- `low_turnover_state`
- `price_volume_divergence`
- `signal_stability`
- 简单的 `orthogonal_residual`

暂缓生成：
- 只改半衰期、窗口、标签口味的微调；
- 与 `momentum_20`、`price_range_position_10`、影线类高度相似的新因子；
- 需要复杂循环 OLS 或大模型黑盒组合器的 proposal。
