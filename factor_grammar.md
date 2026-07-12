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
| `robustness_audit` | 验证集过拟合风险、阶段复核 | `ablate_smoother`, `stress_smoothing_span`, `remove_low_value_layer` | 只允许用在 alpha.py 内的简化/消融式改动；不读取 test，不改 runner |

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

当前 `trade_v4` best 已进入 **Phase 3：稳健性与反过拟合阶段**。
近期真正突破来自：
- #0219 新增 `slow_vol_regime_60`，带来最大结构性提升；
- #0220 删除高相关冗余因子，直接消除同质化惩罚；
- #0250/#0262/#0263 通过自适应平滑和最终 rolling median 提升交易质量。

这意味着继续只调 `span/window/rolling median/ewm` 已经进入尾部收益区。后续 proposal
必须默认把“平滑参数继续加码”视为过拟合风险，而不是主要研究方向。

当前主要瓶颈/风险来自：
- 年度稳定性仍偏弱，但不能只靠继续加平滑解决；
- Top basket Jaccard 已很高、换手已很低，继续降换手的边际价值下降；
- val score 连续由平滑层抬升，存在验证段适配风险；
- 新增因子必须证明低相关、低复杂度，并能保持 #0263 的交易质量。
- 同时需要恢复因子探索积极度：不能因为平滑有效，就停止寻找新的结构性 alpha。

优先生成：
- `factor_pruning`：删掉边际贡献低、与组合高度同质、或只增加复杂度的部件；
- `orthogonal_residual`：只有当 raw idea 与现有组合确有新信息时才进入；
- `low_turnover_state`：慢变量新结构，但不要再只是换一个更长窗口；
- `price_volume_divergence` / `liquidity_shock`：必须控制换手，不破坏 Sharpe；
- `robustness_audit`：在 alpha.py 内做简化/消融式改动，例如移除重复平滑层、检验是否仍保持大部分 score。

暂缓生成：
- 只改半衰期、窗口、标签口味、平滑 span、rolling median window 的微调；
- 继续叠加新的 `ewm`、`rolling median`、`volatility_quantile smoothing`；
- 与 `momentum_20`、`price_range_position_10`、影线类高度相似的新因子；
- 只为了降低换手而牺牲 IC/Sharpe 可解释性的 proposal；
- 需要复杂循环 OLS 或大模型黑盒组合器的 proposal。

## 6. Phase 3 proposal 约束

每组三个候选 proposal 必须满足：

1. 至少两个候选不是 `signal_stability` / `preprocess` 平滑类。
2. 至少一个候选必须是低相关新因子探索，优先 `price_volume_divergence`、`liquidity_shock`、
   `bar_structure` 或 `orthogonal_residual`。
3. 至少一个候选必须是 `factor_pruning`、`orthogonal_residual` 或 `robustness_audit`。
4. 最多一个候选允许涉及平滑，而且必须说明为什么不是简单窗口微调。
5. 如果 proposal 新增因子，必须写明它相对 `slow_vol_regime_60`、`momentum_20`、`price_range_position_10`
   的差异，且预期不提高换手。
6. 如果 proposal 删除或简化组件，必须说明可能损失哪些指标，以及为什么这有助于降低 val 适配风险。

## 7. 因子探索积极度奖励

proposal gate 应奖励“有边界的探索”，不是奖励因子数量。

应加分：
- 使用 `price_volume_divergence`、`liquidity_shock`、`bar_structure`、`orthogonal_residual`
  等当前 best 尚未充分利用的 family；
- 明确说明相对 `slow_vol_regime_60`、`momentum_20`、`price_range_position_10` 的差异；
- 新因子承诺低换手、低复杂度，并且可以替换/删除一个旧部件，而不是无脑追加；
- 使用 residualize/orthogonalize 保留新信息，或先做 raw factor 再做正交残差；
- 针对收益/回撤效率、年度稳定性或超额 Sharpe，而不是只追 IC。

应扣分：
- add_factor 但没有低相关理由；
- 只是同一 primitive 换窗口；
- 与现有慢波动、动量、range position、影线因子高度相似；
- 新因子会明显增加换手、复杂度或模型黑盒程度。
