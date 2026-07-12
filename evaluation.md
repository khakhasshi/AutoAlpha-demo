# AutoAlpha — 评估宪法（Evaluation Constitution）· trade v4

> **本文件是人类维护的"评估锁定"** — agent 不可修改。
> 所有"主分如何计算 / 标签如何定义 / 回测如何模拟"的真理来源在 `prepare.py`，本文件是其规约说明。
> 当且仅当人类完成一轮 `judge.py` 测试集复核后，可主动迭代本文件。

---

## 1. 主分（Primary Score）· trade v4 奖励机制版 — runner 的 accept / revert 唯一依据

**v4 设计动机**：trade v3 已经把当前最高分承接到更交易化的评价体系，但 agent 仍可能
继续优化单一总分的缝隙。v4 把 reward 从“结果分”扩展为“结果质量 + 研究行为约束”：
既奖励超额质量、收益/回撤效率和年度稳定性，也惩罚高换手、过多因子、过长代码和高同质化。

```
score = 0.22 * rank_ic_ir
      + 2.50 * rank_ic_mean
      + 1.20 * pearson_ic_mean
      + 1.65 * clip(excess_sharpe, -3, 3)
      + 1.10 * clip(sharpe, -3, 3)
      + 2.80 * clip(excess_annual_return, -0.50, 0.50)
      + 1.40 * clip(annual_return, -0.50, 0.50)
      + 1.20 * clip(excess_annual_return / max(abs(excess_max_drawdown), 0.10), -2, 2)
      + 0.90 * clip(annual_return / max(abs(max_drawdown), 0.10), -2, 2)
      + 0.75 * clip(1 + excess_max_drawdown / 0.30, -2, 1)
      + 0.75 * clip(1 + max_drawdown / 0.50, -2, 1)
      + 0.45 * monotonicity
      + 0.50 * year_stability
      + 0.25 * positive_year_ratio
      - 0.55 * clip(annual_turnover / 40 - 1, 0, 4)
      - 0.20 * clip((factor_count - 18) / 10, 0, 3)
      - 0.15 * clip((alpha_lines - 700) / 500, 0, 3)
      - 0.35 * clip((max_factor_corr - 0.75) / 0.15, 0, 3)
```

分量含义：
- **IC 底座**：`rank_ic_ir / rank_ic_mean / pearson_ic_mean` 仍然奖励排序预测能力，但权重继续降低。
- **超额质量主项**：`excess_sharpe` 与 `excess_annual_return` 权重继续上升，减少市场 beta 污染。
- **收益/回撤效率**：新增 `excess_efficiency` 与 `return_efficiency`，奖励同等回撤下更高收益。
- **绝对多头质量**：`sharpe` 与 `annual_return` 防止模型只会相对跑赢但绝对亏损严重。
- **回撤质量**：`1 + drawdown / threshold` 在 0 回撤时给正分，在阈值附近归零，深回撤给负分。
- **年度稳定性**：`year_stability` 与 `positive_year_ratio` 奖励验证段内分年表现更均衡的策略。
- **单调性**：保留分组收益单调性的奖励，但权重低于收益质量。
- **换手惩罚**：年换手超过 40 后开始扣分，抑制过度响应近期 IC 的解。
- **复杂度惩罚**：因子数超过 18、代码超过 700 行后逐步扣分。
- **同质化惩罚**：相关性门控提供的 `max_factor_corr` 超过 0.75 后逐步扣分。

阈值解释：
- `max(abs(excess_max_drawdown), 0.10)`：超额效率至少按 10% 回撤预算折算，避免 0 回撤除法爆炸。
- `max(abs(max_drawdown), 0.10)`：绝对收益效率同理。
- `excess_max_drawdown / 0.30`：超额回撤 30% 是一个重要警戒线。
- `max_drawdown / 0.50`：绝对多头回撤 50% 后不应再被高 IC 掩盖。
- `annual_turnover / 40 - 1`：年单边换手 40 以内暂不惩罚，超过后逐步扣分。
- `factor_count - 18`：18 个以内的因子不惩罚，超过后要求证明边际价值。
- `alpha_lines - 700`：700 行以内不惩罚，超过后抑制复杂黑盒。

实现位置：`prepare.primary_score()` — **唯一来源**。
agent 不得镜像、重写、绕过；`runner.py` 永远只读 `score` 这一个数字判断 accept / revert。

### 取舍

trade v4 不保证每个 accepted 因子都能直接实盘，但它会显著降低以下坏解被接受的概率：

- 只靠 rank IC 拿高分，但多头年化和超额年化很差；
- 只靠回撤质量固定项拿高分，但收益/效率没有继续改善；
- 高频繁调仓导致换手偏高；
- 不断堆叠高度相关因子；
- 用复杂代码或脆弱模型换取微小分数；
- 回撤极深但 IC 漂亮；
- Top 组合不赚钱，只是相对跌得少。

---

## 2. HORIZON（持有期）— 单一真理来源

允许值：`(1, 3, 5, 10, 20)` 交易日。`alpha.HORIZON` 一旦设定，
`prepare.make_labels` 与 `prepare.backtest` 必须用同一个 H。
错配（label 用 H=5、回测用 H=10）一律视为作弊，runner 校验拒绝。

---

## 3. LABEL_KIND（标签口味菜单）

| kind | 公式 | 用途 |
|---|---|---|
| `raw` | `open[T+1+H] / open[T+1] - 1` | 看绝对收益方向 |
| `market_neutral` | `raw - mean(raw, axis=1)` | **推荐**：扣除当日截面等权市场收益 |
| `vol_adjusted` | `raw / std_20(daily_return)` | 高波动股惩罚，更稳健 |
| `rank` | 截面分位 ∈ [0, 1] | 用 rank 作回归目标 |
| `zscore` | 截面 z-score | 与 rank 类似，保留尾部信息 |

实现位置：`prepare.make_labels(panel, horizon, kind=...)`。
所有标签口味**只影响 IC 计算**，**不影响回测的真金白银模拟**（回测永远是 T+1 开盘买入、Top 10% 等权、双边 15bp、涨停板/停牌剔除）。

---

## 4. 双 IC 指标

`prepare.compute_rank_ic` 与 `prepare.compute_pearson_ic` **同时计算并暴露**：

| 指标 | 算法 | 解读 |
|---|---|---|
| **rank IC** | 截面 rank 后 Pearson | 对极端值稳健；进入主分 |
| **Pearson IC** | 截面原始值 Pearson | 捕捉强度信息；进入主分但权重较低 |

**对照看的研究信号**：
- rank IC 高、Pearson IC 低 → 因子**排序对，量级失真**（如有少数极端值主导）
- Pearson IC 高、rank IC 低 → 因子**被极端值主导**，整体相关性弱

trade v4 中，IC 指标仍然进入主分，但不再是唯一目标。Agent 必须同时关注回测质量、
超额表现、回撤和换手。

---

## 5. 诊断指标（agent 可在 `metrics.py` 自由扩展）

诊断指标中未列入 §1 公式的部分**不影响 accept / revert**，仅供：
- 写入 `journal/runs.jsonl` + 因子卡，给人类回看
- 给 agent 自己在下一轮迭代时参考

**白名单方向**：rank IC、Pearson IC、IC_IR、IC decay（不同 H）、分位数收益、
因子相关性、按市值/波动分组的子集 IC、换手分解、Top 篮子 Jaccard。

**禁止方向**：偷偷加权再 return；偷看 test 段（`AUTOALPHA_TEST_LOCKED` 锁会拦）。

---

## 6. 信号格式硬约束（违者 runner 直接判失败）

`alpha.run` 返回的 `(signal_train, signal_val)` 必须满足：

| 约束 | 阈值 |
|---|---|
| shape | `[date × symbol]` 二维 DataFrame |
| index | `pd.DatetimeIndex` |
| dtype | float |
| 截面规模 | 每日列数 ≥ 30 |
| 截面均值 | `mean(\|daily_mean\|) < 0.05` |
| 截面标准差 | `0.5 < mean(daily_std) < 2.0` |
| 全 NaN 行占比 | < 50% |

实现位置：`prepare.validate_signal`。
这是**截面研究的硬门禁**：agent 必须每天对信号做 z-score 或 rank 后才能交。

---

## 7. 多头回测定义（人类锁定）

| 项 | 规则 |
|---|---|
| 选股 | 每日按 signal 截面排名取 Top 10% |
| 入场 | T+1 开盘 |
| 出场 | T+1+H 开盘 |
| 持仓 | 等权 |
| 重叠持仓 | 每日 1/H 资金被换仓 |
| 不可买 | `trade_status==1` 或 `close >= limit_up*0.99` |
| 不可卖 | `trade_status==1` 或 `close <= limit_down*1.01` |
| 手续费 | 双边 15 bp |

### 7.1 Benchmark 与超额指标

**数据**：`stock_data/benchmark_852_all.parquet` — 中证 1000 指数日 close。
2016-01-04 → 2026-06-04，2529 个交易日，由 `prepare.load_benchmark_series()` 公开。

**对齐**：所有 prepare 内部的 `_pivot` 现在自动 reindex 到 `trading_Calendar.parquet`
中 `is_trade==1` 区间内的交易日。这意味着 `shift(-1)` 严格按"下一个交易日"语义工作，
不会因 panel 个别日期缺失而跨假期错位。

**超额定义（加性）**：
```
excess_daily = portfolio_daily − benchmark_daily
excess_nav   = (1 + excess_daily).cumprod()
excess_dd    = excess_nav / excess_nav.cummax() − 1.0
excess_sharpe        = excess_daily.mean() / excess_daily.std() * √245
excess_annual_return = excess_nav[-1]^(245/n_days) − 1
excess_max_drawdown  = excess_dd.min()
```

**ScoreReport 超额字段**（trade v4 已进入 score 公式）：
- `excess_annual_return`
- `excess_sharpe`
- `excess_max_drawdown`

**runner / judge / factor_library 联动**：
- runner 的 `runs.jsonl` 与 `journal/notes/{iter}.md` 都会带这三个字段
- factor_library 的 `chart_overview.png` 升级为 6 联图：
  ①多头净值 ②累计 IC ③10 分组日均 ④多头回撤 ⑤超额收益（多头/基准/超额三线）⑥超额回撤
- judge.py 现在在 `journal/test_charts/<ts>_<F00xx>_<name>/` 下生成 test 与 val 两张同款 6 联图



---

## 8. 数据切分（永久冻结）

```
train: 2016-01-04 → 2021-12-03    (~5.9 年)
val:   2021-12-04 → 2024-12-03    (3.0 年)
test:  2024-12-04 → 2026-06-04    (1.5 年)
```

任何对 splits.json 的人为修改、对 parquet 文件的替换都会被 checksum 校验抓出。

---

## 9. trade v4 说明

**版本**：trade_v4

**主分公式**：`IC 底座 + 超额质量 + 收益/回撤效率 + 年度稳定性 + 回撤质量 - 换手/复杂度/同质化惩罚`（详见 §1）

**评估方式**：固定 train / val / test 三段切分（详见 §8）。训练在 train 段，
Agent accept/revert 依据 primary_score 在 val 段的分数。test 段由 `judge.py`
在人类主动触发时才解锁。

**研究方向**：
- 优先寻找新的、低相关的因子族，而不是继续微调 IC 权重半衰期；
- 允许牺牲一部分 IC_IR 来换取更高的超额收益/回撤效率；
- 鼓励降低换手、提高年度稳定性、Top 篮子稳定性与收益质量；
- 不鼓励继续堆叠只提升 IC、但恶化交易质量或可解释性的高换手因子；
- 新因子若高度同质，应先做残差化或换研究方向。

**Score 公式的合理性由 §13 异常熔断机制兜底**（详见 `program.md` §13）：
若 score REJECTED 但底层指标显著改善（Sharpe > +30% / MDD 改善 > 20% / 年化 > +30%），
runner 会打出 `⚠ SCORE_ANOMALY` 标记，提示人类介入检查公式设计。
