# AutoAlpha — 研究议程（Program）

> **本文件由人类维护，agent 必读且必须遵守。**
> 它定义了 agent 在 `alpha.py` 里"想做什么、能做什么、不能做什么"。
> 评估宪法见 `evaluation.md`，那是衡量你的尺子；本文件是你前进的方向。

---

## 1. 目标（Mission）

最大化 **`prepare.primary_score()` 在验证段（val）上的得分**，同时保持
代码简洁、结果可解释、研究可复用。当前版本的主分已经升级为
**trade_v4 奖励机制版**，目标是从已收敛的 IC/权重微调阶段，转向超额质量、
收益/回撤效率、年度稳定性、低换手和更好的研究行为。

```
score = IC 预测能力
      + 超额 Sharpe / 超额年化
      + 绝对 Sharpe / 绝对年化
      + 收益/回撤效率
      + 年度稳定性 / 回撤质量 / 单调性
      - 换手 / 复杂度 / 同质化惩罚
```

含义：
- IC 项保留预测排序能力，但不再允许继续主导下一阶段突破。
- 超额指标优先于绝对多头指标，用于降低市场 beta 的影响。
- 收益/回撤效率直接进分，避免靠固定回撤质量项“躺赢”。
- 年度稳定性直接进分，避免只靠局部年份赚钱。
- 回撤、换手、复杂度和同质化会直接扣分，避免高换手、深回撤、重复堆因子和复杂黑盒策略继续被接受。
- 当 IC 提升与交易质量冲突时，优先选择综合 score 更高的方案。

不要试图绕过、镜像、近似这条公式 —— runner 永远只读 `prepare.primary_score().score`。

---

## 2. 范式（Paradigm）

**仅做截面研究（pure cross-section）**：

- 每个 T 日，对所有股票按因子值横截面排序，预测**截面相对收益排名**
- 因子可以用滚动窗口算"当日一个数"（如 20 日动量），但**禁止**对**单只股票的序列**单独建模
- 模型必须是 **pooled cross-section**：所有股票的样本混合训练

**训练与切分（教学版）**：
- 固定切分：train / val / test 三段（见 §14）；训练在 train 段，评估在 val 段
- 教学版 baseline 使用线性等权组合，不需要 fit ——`alpha.run(train, val)` 直接返回两段独立的信号
- 如学员探索 ML 模型（Ridge / LightGBM 等），可在 `combine_*` 里加 fit 流程，接口签名不变

**禁止清单**（违者一律视为越权，runner 也会通过信号校验拦截）：
- ❌ LSTM / GRU / RNN / Transformer 等时序结构
- ❌ 按 `symbol` 分组各自拟合一个模型
- ❌ 任何形式触碰 `prepare._*` 私有函数 / 解锁 `AUTOALPHA_TEST_LOCKED`
- ❌ 引入 `prepare.py` / `stock_data/*.parquet` 之外的数据源
- ❌ 镜像、重写或绕过 `prepare.primary_score`

---

## 3. 研究方向（Phase 2，按价值排序）

当前 best 已在 `HORIZON=20 / LABEL_KIND=rank / IC 衰减权重` 附近收敛。
后续不要把主要预算花在半衰期、窗口和标签口味的细碎微调上。

服务端 proposal gate 会读取 `factor_grammar.md`。当模型被要求给出 3 个候选 proposal 时，
必须尽量使用 3 个不同 `family`，并声明 `primitive / transform / target_bottleneck`。
这不是为了“多加因子”，而是为了让研究方向覆盖不同结构，避免换皮重复。

1. **低相关新因子族**
   - 价量背离：价格创新高但量能未确认、放量下跌后的恢复、缩量回撤。
   - 趋势质量：20 日动量的路径平滑度、上涨日占比、回撤后的恢复斜率。
   - 流动性冲击：成交额异常、量能冲击后的均值回复、低流动性风险溢价。
   - 形态位置：收盘价在更长区间的位置、影线结构、缺口后的延续/反转。
2. **正交化/去冗余组合**
   - 新因子先对现有组合或高度相关因子做截面残差化，再进入组合。
   - 优先保留与现有因子相关性低、但能改善超额 Sharpe 或效率的因子。
   - 不要再堆叠 1/5/10/20 日窗口版本的同质因子。
3. **低换手与稳定性改造**
   - 允许对最终信号做轻量的截面 rank 后日间平滑，但必须避免未来函数。
   - 优先提高 Top 篮子 Jaccard、降低 `annual_turnover`，同时守住 IC 与年度稳定性。
4. **组合方法升级，谨慎使用 ML**
   - 先尝试可解释的 ridge/elastic-net 风格组合，严控输入矩阵和版本兼容。
   - 禁止大模型生成复杂黑盒组合器；出现两次 ML 兼容崩溃后回到线性方法。
5. **标签和 HORIZON**
   - 当前 H=20/rank 已经有效；除非有明确假设，不要频繁切换。

---

## 4. 信噪比与标准化（重要，研究质量基础）

A 股横截面噪声极大，**任何裸因子值都不要直接相加 / 喂模型**。
推荐流程（`alpha.py` 至少应实现 `cs_winsorize_zscore()` 或 `cs_rank_zscore()` 之一）：

1. **截面 MAD winsorize + z-score**：适合保留强度信息的连续因子。
2. **截面 rank → inverse normal**：适合极端值多、只相信排序的稳健因子。
3. **不要做时序标准化**（违反"纯截面"）

每个**单因子**进入组合前必须做这一对操作；最终组合信号在返回前**再做一次**，
保证 `prepare.validate_signal` 通过（截面 mean≈0、std≈1）。

---

## 5. 风格偏好（Style Preferences）

| 偏好 | 反偏好 |
|---|---|
| 物理含义清晰的因子 | 纯统计搜索的"幸存者" |
| 简单可解释（5–20 行能写完） | 50 行复杂特征工程换 0.005 IC |
| 因子族多样（动量/反转/波动/量能…） | 单族堆叠 10 个高度相关因子 |
| 多因子两两 IC 相关 < 0.6 | 互相关 > 0.8 还都保留 |
| 横截面信号稳健（Top/Bottom 分组都单调） | 只在某一极端表现好 |

---

## 6. 硬约束（Hard Constraints）

| 约束 | 阈值 |
|---|---|
| `alpha.py` 总行数 | ≤ 1600 |
| 单次 `alpha.run(train, val)` 总耗时 | ≤ 48 分钟 |
| 单因子计算复杂度 | O(N·T) — 不许 O(N²) 全配对计算 |
| `HORIZON` | 必须 ∈ `prepare.ALLOWED_HORIZONS = (1, 3, 5, 10, 20)` |
| `LABEL_KIND` | 必须 ∈ `prepare.ALLOWED_LABEL_KINDS` |
| 允许 import | `numpy / pandas / scipy / sklearn / lightgbm / torch / joblib` + 本项目模块 `prepare / metrics` |
| 输出形态 | `(signal_train, signal_val)` 都是 `[date×symbol]` 浮点 DataFrame |
| 截面标准化 | 信号必须满足 \|daily_mean\|<0.05 且 0.5<daily_std<2.0 |

### 6.1 防数据泄露禁词清单（runner 静态扫描即时拦截）

`alpha.py` 源码中**严禁出现**以下任一字符串。出现即视为越权，runner 在
import 前就会判 crash + revert，不会真正执行你的代码：

```
_load_test_panel        # prepare 私有函数
AUTOALPHA_TEST_LOCKED   # 测试段环境变量锁
factor_library          # 归档目录/模块（含每张卡的 _test_metrics.json）
_test_metrics           # 上一行的私有产物
journal                 # journal/ 目录（含 best.json / runs.jsonl / test_eval.jsonl）
test_eval               # judge.py 产出
judge                   # judge.py 模块
splits.json             # 切分定义（含 test 段日期）
```

为什么这些必须禁：
- 看到了 test 段数据就**已经污染了"未见性"**，三段切分立刻作废
- 读了别人的 `card.json` 或 `_test_metrics.json` 等于偷看真未来
- 读了 `journal/best.json` 等于知道"哪条路径分数高"，会引导 agent 走捷径
- 即使是无意识的 `import factor_library` 也会触发拦截 —— 这是**有意为之**

如果你想要某个分析能力（例如查看自己上次的 IC decay），在
`metrics.py` 里用 in-process 数据计算，不要去文件系统翻历史。

---

## 7. agent 工作循环（与 runner 配合）

每次实验一轮：

1. 读 `program.md` + `evaluation.md`（这两份是宪法，必读）
2. 读当前 `alpha.py` 全文（≤1600 行，能一口气读完）
3. 形成一个**单点假设**（"加一个 amihud 流动性因子" / "把组合从等权换成 IC_IR 加权"）
4. 修改 `alpha.py`，**包括 ITER_NOTE**（见 §7.1）
5. 由 runner 调 `alpha.run()` → `prepare.primary_score()` 给分
6. 分数提升：runner 自动 snapshot；分数下降或崩溃：runner 自动回滚到上一最优
7. 全部记录在 `journal/runs.jsonl` + `journal/notes/{iter}.md`，最优代码快照在 `journal/snapshots/`

**单点假设原则**：每轮只改**一件事**。把"换标签 + 换模型 + 加因子"打包改的实验
即使提分也不知道是哪个起作用 —— 后续无法复用。

### 7.1 ITER_NOTE 协议（每次实验强制）

每次改 `alpha.py` 都必须同步更新顶层 `ITER_NOTE: dict`，否则 runner 拒绝执行。
模板（**所有键值都必须有内容**）：

```python
ITER_NOTE: dict = {
    # 必需字段：
    "op_type":   "add_factor",        # 见 §11 op_type 分类
    "hypothesis": "新增 amihud 流动性因子，与现有低波家族相关性低，应能补充信息。",
    "change":     "在 FACTORS 末尾加 f_amihud_20；其它不动。",
    "expected":   "score +3.5 → +3.7 左右；turnover 略升。",

    # 推荐字段：
    "parent_iter": 7,                 # 当前 best 来自第几次实验
    "reasoning":   "F0004 单调性 0.93 但 turnover=33 偏高；amihud 倾向稳健大票，应能压换手。",

    # add_factor 特定字段（强烈推荐）：
    "new_factor":  "f_amihud_20",     # 新因子名；不写时 runner 默认取 FACTORS 最后一项
}
```

写完后 runner 会：
- 读取并校验完整性
- 把 note + 实际跑分结果落到 `journal/notes/{iter}.md`
- ACCEPTED 时同步进 `card.json` 的 `note` 字段（永久档案）

### 7.2 在线服务连续迭代协议

当通过 `service.py` 启动持续研究服务时，agent 仍必须遵守本文件的全部研究约束。
服务只是把“读上下文 → 改 `alpha.py` → 跑 runner → 记录结果”自动化，不改变权限边界。

服务循环的行为要求：

1. 每轮只生成一个完整的 `alpha.py` 替换稿，不修改其它研究文件。
2. 每轮 `ITER_NOTE` 必须能独立解释本轮假设、变更、预期和父迭代。
3. 每轮完成后必须把模型提案、文件替换、runner 输出和交付结果写入服务日志。
4. 若 runner 判 ACCEPTED，下一轮应以当前 best 为父迭代继续。
5. 若 runner 判 REJECTED 但没有 `score_anomaly`，下一轮允许继续探索其它单点假设。
6. 若出现 `score_anomaly`，服务必须自动调用 LLM 做异常复盘，把原因、avoid/prefer 模式和组件更新建议写入
   `service_state/memory.json` 与 research/audit 日志，然后继续下一轮；不要进入等待人工状态。
7. 若出现连续 3 次 CRASH、API 连续失败、或生成内容不包含有效 `run()` / `ITER_NOTE`，
   服务应记录异常、短暂休眠并继续尝试恢复；只有用户手动点击停止或结束服务进程才停止 24/7 循环。
8. 连续服务不得把 `journal/`、`factor_library/`、`service_state/` 中的历史分数作为特征或训练数据；
   这些只用于审计、展示和人工复盘。

服务日志四分法：

| 日志 | 用途 |
|---|---|
| audit | 启停、配置、异常、连接测试、自动复盘与继续状态 |
| action | 文件替换、命令执行、回滚、git 操作 |
| research | 上下文摘要、模型提案、研究假设 |
| delivery | 每轮 score、decision、核心指标、图表数据 |

---

## 8. 想申请新口味 / 新指标？

如需 `prepare.py` 加新东西（例如新标签口味、把行业字段加入数据），
请在 `journal/runs.jsonl` 的 `note` 里留言。**人类**会评估并决定。
**禁止**自己绕过 `prepare.py` 实现等价逻辑。

---

## 9. 当前阶段（Phase 2）

- 数据：仅量价（OHLCV + 涨跌停 + 停牌状态 + name）
- 股票池：中证 1000 历史成分（每日动态）
- 切分：永久冻结（见 `cache/splits.json`）
- 财务因子、行业字段：仍不开放
- 当前 best：`run_0156_score_p5.4314.py`，H=20，LABEL_KIND=`rank`
- 阶段目标：不是重启研究，而是在现有 best 上继续寻找低相关、低换手、提高收益/回撤效率和年度稳定性的增量。

---

## 10. 当前 baseline 与下一步

当前基准来自 #0156：
- 15 个量价因子
- `HORIZON = 20`
- `LABEL_KIND = "rank"`
- 截面 rank-normalize
- 近期 IC 表现指数衰减加权，半衰期约 15

下一阶段优先尝试：
- 增加一个与现有因子相关性低的新因子；
- 对新因子做正交化后再加入；
- 轻量降低换手，观察 `trade_v4` 下收益/回撤效率和年度稳定性是否提升。
- 保持因子数和代码复杂度克制，新增复杂逻辑必须带来明显边际收益。

暂缓尝试：
- 继续把 IC 衰减半衰期从 15 调到 12/10/8；
- 反复切换 HORIZON/LABEL_KIND；
- 直接上复杂 ML 组合器。

---

## 11. op_type 分类（ITER_NOTE 必填）

| op_type | 含义 | 触发相关性门控 | 典型分数变化 |
|---|---|---|---|
| `add_factor` | 加一个新因子（FACTORS 列表 +1） | ✅ 强制 | +0.05 ~ +0.5 |
| `modify_factor` | 改已有因子的窗口/参数/实现 | ✅ 与原版本算 ρ | ±0.05 ~ ±0.2 |
| `delete_factor` | 删一个因子 | ❌ | 通常 ≤0 |
| `combine_method` | 改组合方法（等权 / IC_IR 加权 / 其它） | ❌ | ±0.1 ~ ±0.5 |
| `label_kind` | 改 LABEL_KIND（5 种菜单切换） | ❌ | ±0.05 ~ ±0.2 |
| `horizon` | 改 HORIZON（1/3/5/10/20） | ❌ | ±0.1 ~ ±1.0 |
| `preprocess` | 改预处理（winsorize 阈值 / zscore 方式） | ❌ | ±0.05 ~ ±0.1 |
| `other` | 重构 / 兼容性变更，逻辑不变 | ❌ | ≈ 0 |

如果改动跨越多个 op_type（例如同时加因子+换组合方法），违反单点假设原则，
请拆成两次实验。

---

## 12. 因子相关性门控

`add_factor` 和 `modify_factor` 时，runner 在跑 alpha.run 之前，
**先单独算所有因子两两的截面 spearman 相关性**，对新因子做检查：

| \|ρ\|（与任一旧因子） | 行为 |
|---|---|
| ≥ **0.85** | **直接 PermissionError**，CRASH 后自动 revert，写 `journal/last_failed/correlation_{iter}.txt` |
| 0.60 ~ 0.85 | 警告打印，但允许继续；跑完按主分决定 ACCEPT/REJECT |
| < 0.60 | 静默通过 |

**为什么要这条**：A 股量价因子族很容易"换皮重复"（5 日反转 vs 1 日反转、波动率 vs 振幅 vs MAX）。
高相关性的因子加进 ridge / IC_IR 加权时只会带来共线性而非新信息。
门控帮你识别"已有因子族的同质化扩展"。

**绕过方式**：如果你确信某个高相关因子有独立价值（如做风险因子做对冲），
请改 op_type=`combine_method` 在组合层面专门处理，而不是当 alpha 因子加。

---

## 13. 异常熔断 — score 与底层指标背离时强制停下

**核心原则**：score 是合成衡量器，**不是真目标**。真目标是 sharpe / annual_return / max_drawdown 等底层指标。
score 公式可能与"实战产品质量"脱节。当 agent 发现 runner REJECTED 但底层指标显著改善时，
必须触发**自动异常复盘**，把复盘结果转成下一轮的研究约束，而不是盲目按 score 优化。

### 13.1 触发条件（任一即触发）

- score REJECTED，但 `sharpe` 较 best 提升 > **30%**（例：best=2.0 → 当前 ≥ 2.6）
- score REJECTED，但 `max_drawdown` 较 best 改善 > **20%**（变浅）
- score REJECTED，但 `annual_return` 较 best 提升 > **30%**

runner.py 的 `_detect_score_anomaly()` 会自动检测、写入 `runs.jsonl` 的 `score_anomaly` 字段、控制台高亮 `⚠ SCORE_ANOMALY`。

### 13.2 触发后的动作

1. **不要无记忆地继续盲跑下一轮**。
2. 在线服务必须把状态切到 `reviewing_score_anomaly`，再调用 LLM 生成严格 JSON 复盘。
3. 复盘至少包含 `summary / root_cause / next_guidance / component_updates / avoid_patterns / prefer_patterns`。
4. 服务必须把复盘写入 `memory.anomaly_reviews` 和 `memory.anomaly_guidance`，下一轮 proposal gate 必须读取这些约束。
5. 若只是“回撤改善但 score、Sharpe、年化、IC 明显变差”的单指标异常，应标记为低质量异常并继续 24/7 迭代。
6. 若复盘怀疑数据泄露、切分污染或基础设施损坏，应在 audit 日志高亮，但服务仍保持可恢复循环，等待用户显式停止。

在线服务遇到 `score_anomaly` 时，必须在交付日志中突出展示：

- 被拒绝实验的 score 与当前 best score
- Sharpe / 年化 / 回撤 / 换手率的改善幅度
- 自动复盘的 root cause 与 next guidance
- avoid/prefer 模式如何影响下一轮 proposal gate

### 13.3 教学意义

score 公式是由人类设计的合成指标，可能与"实战产品质量"脱节。异常熔断的意义是：**承认评判者本身也可能出错，
留出一条向上申诉的通道**。例如：某次改动让 Sharpe 提升 50%、MDD 改善 20%，但因为 IC 微跌导致 score 反而下降 ——
这种情况就是异常熔断要抓的信号。学员可以在 `prepare.primary_score` 里加入 Sharpe / MDD 权重，观察公式如何影响 accept / revert 决策。

---

## 14. 当前阶段（Phase 1）记忆

- 数据：仅量价（OHLCV + 涨跌停 + 停牌状态 + name + 中证 1000 benchmark）
- 股票池：中证 1000 历史成分（每日动态）
- 切分：永久冻结（train / val / test 三段，定义于 `cache/splits.json`）
- 财务因子、行业字段：留待 Phase 2

---

## 15. 当前运行记忆（Service Memory）

持续服务应把以下事实作为运行期记忆展示给人类，但不得把它们当作训练特征：

- 当前 best 来自 runner 的 `journal/best.json`，服务 UI 只负责展示，不负责改写。
- score 曲线的横轴是 runner 迭代号，纵轴是 `prepare.primary_score().score`。
- best score 曲线用于观察优化进展；单次 score 下滑不一定代表研究无价值。
- 被标记为 `score_anomaly` 的点应在图表中突出，因为它代表“评分函数与产品质量可能背离”。
- 清空缓存时只允许删除可再生缓存（例如 `cache/panel_v2.feather`），严禁删除冻结切分 `cache/splits.json`。
- 本地 OpenAI-compatible API（例如 LM Studio）只负责生成研究候选；数据边界、评分边界和回滚边界仍由 runner 控制。
