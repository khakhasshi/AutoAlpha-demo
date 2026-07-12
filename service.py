"""
AutoAlpha local service.

Runs a small browser UI plus a background research loop. The loop uses an
OpenAI-compatible /v1/chat/completions endpoint to propose complete alpha.py
updates, evaluates them with runner.py, and records audit/action/research/
delivery logs under service_state/logs/.
"""
from __future__ import annotations

import html
import json
import math
import os
import subprocess
import threading
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "service_state"
LOG_DIR = STATE_DIR / "logs"
CONFIG_PATH = STATE_DIR / "config.json"
MEMORY_PATH = STATE_DIR / "memory.json"
ALPHA_PATH = ROOT / "alpha.py"
PYTHON = ROOT / ".venv" / "bin" / "python"
RUNNER = ROOT / "runner.py"

LOG_FILES = {
    "audit": LOG_DIR / "audit.jsonl",
    "action": LOG_DIR / "action.jsonl",
    "research": LOG_DIR / "research.jsonl",
    "delivery": LOG_DIR / "delivery.jsonl",
}

DEFAULT_CONFIG: dict[str, Any] = {
    "base_url": "",
    "api_key": "",
    "model": "",
    "temperature": 0.2,
    "iteration_sleep_sec": 5,
    "auto_commit_accepted": False,
    "memory_enabled": True,
    "auto_review_score_anomaly": True,
    "enabled": False,
}

ALLOWED_PROPOSAL_FAMILIES = {
    "path_quality",
    "low_turnover_state",
    "price_volume_divergence",
    "liquidity_shock",
    "bar_structure",
    "orthogonal_residual",
    "factor_pruning",
    "signal_stability",
    "robustness_audit",
}

STATE_LOCK = threading.RLock()
LOOP_THREAD: threading.Thread | None = None
RUNTIME: dict[str, Any] = {
    "running": False,
    "stop_requested": False,
    "iteration": 0,
    "status": "idle",
    "last_error": None,
    "last_started_at": None,
    "last_finished_at": None,
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_state() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for path in LOG_FILES.values():
        path.touch(exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
    if not MEMORY_PATH.exists():
        MEMORY_PATH.write_text(json.dumps(default_memory(), indent=2), encoding="utf-8")


def default_memory() -> dict[str, Any]:
    try:
        import prepare
        score_version = getattr(prepare, "SCORE_VERSION", "demo_v1")
    except Exception:
        score_version = "demo_v1"
    return {
        "created_at": now_iso(),
        "updated_at": None,
        "score_version": score_version,
        "summary": "No service iterations have been memorized yet.",
        "best_known": None,
        "avoid": [],
        "promising": [],
        "recent": [],
        "anomaly_reviews": [],
        "anomaly_guidance": "",
        "stage_guidance": "Phase 3: avoid more smoothing parameter mining; keep an explicit exploration budget for low-correlation structural factors, especially price-volume divergence, liquidity shock, bar structure, and orthogonal residuals.",
    }


def load_config(include_secret: bool = True) -> dict[str, Any]:
    ensure_state()
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        cfg = {}
    out = {**DEFAULT_CONFIG, **cfg}
    if not include_secret and out.get("api_key"):
        out["api_key"] = "********"
    return out


def save_config(data: dict[str, Any]) -> dict[str, Any]:
    current = load_config()
    cfg = {**current}
    for key in DEFAULT_CONFIG:
        if key in data:
            cfg[key] = data[key]
    if data.get("api_key") == "********":
        cfg["api_key"] = current.get("api_key", "")
    cfg["temperature"] = float(cfg.get("temperature") or 0.2)
    cfg["iteration_sleep_sec"] = max(0, int(cfg.get("iteration_sleep_sec") or 0))
    cfg["auto_commit_accepted"] = bool(cfg.get("auto_commit_accepted"))
    cfg["memory_enabled"] = bool(cfg.get("memory_enabled", True))
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    append_log("audit", "config_saved", {
        "base_url": cfg.get("base_url"),
        "model": cfg.get("model"),
        "has_api_key": bool(cfg.get("api_key")),
        "iteration_sleep_sec": cfg.get("iteration_sleep_sec"),
        "auto_commit_accepted": cfg.get("auto_commit_accepted"),
        "memory_enabled": cfg.get("memory_enabled"),
    })
    return cfg


def append_log(kind: str, event: str, payload: dict[str, Any]) -> None:
    ensure_state()
    rec = {
        "ts": now_iso(),
        "kind": kind,
        "event": event,
        "payload": payload,
    }
    with LOG_FILES[kind].open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def read_log(kind: str, limit: int = 200) -> list[dict[str, Any]]:
    ensure_state()
    lines = LOG_FILES[kind].read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except Exception:
            out.append({"ts": "", "kind": kind, "event": "parse_error", "payload": {"raw": line}})
    return out


def load_memory() -> dict[str, Any]:
    ensure_state()
    try:
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = default_memory()
    base = default_memory()
    base.update(data)
    try:
        import prepare
        current_score_version = getattr(prepare, "SCORE_VERSION", "demo_v1")
    except Exception:
        current_score_version = "demo_v1"
    if base.get("score_version") and base.get("score_version") != current_score_version:
        seeded = seed_memory_from_journal(default_memory())
        save_memory(seeded)
        return seeded
    current_runs = [r for r in journal_runs(limit=200) if r.get("score_version", "demo_v1") == current_score_version]
    has_memory_items = bool(base.get("best_known") or base.get("recent") or base.get("avoid") or base.get("promising"))
    if not current_runs and has_memory_items:
        seeded = seed_memory_from_journal(default_memory())
        save_memory(seeded)
        return seeded
    if not base.get("recent") and not base.get("updated_at"):
        seeded = seed_memory_from_journal(base)
        if seeded.get("recent"):
            save_memory(seeded)
            return seeded
        if seeded.get("summary") != base.get("summary"):
            save_memory(seeded)
            return seeded
    best_rec = current_best_record()
    if best_rec and (
        not base.get("best_known")
        or float(best_rec.get("score") or float("-inf")) > float((base.get("best_known") or {}).get("score") or float("-inf"))
    ):
        base["best_known"] = {
            "runner_iter": best_rec.get("iter_id"),
            "score": best_rec.get("score"),
            "summary": "Synced from journal/best.json current baseline.",
            "factor_library": best_rec.get("factor_library"),
        }
        base["summary"] = (
            f"Synced to current baseline runner#{best_rec.get('iter_id')} "
            f"score={best_rec.get('score')}; continue 24/7 iteration from this best."
        )
        save_memory(base)
        return base
    if not base.get("anomaly_reviews"):
        anomalies = [
            {
                "runner_iter": r.get("iter_id"),
                "score": r.get("score"),
                "decision": r.get("decision") or r.get("status"),
                "score_anomaly": r.get("score_anomaly"),
                "summary": "Seeded historical score_anomaly; future events are auto-reviewed instead of pausing.",
            }
            for r in current_runs
            if r.get("score_anomaly")
        ][-8:]
        if anomalies:
            base["anomaly_reviews"] = anomalies
            base["anomaly_guidance"] = (
                "Historical score_anomaly events exist. In 24/7 mode, treat isolated drawdown improvement "
                "as insufficient when score, Sharpe, return or IC degrade materially; auto-review and continue."
            )
    if "pause" in str(base.get("summary", "")).lower() or "等待" in str(base.get("summary", "")):
        best = base.get("best_known") or {}
        base["summary"] = (
            f"24/7 mode active from baseline runner#{best.get('runner_iter')} score={best.get('score')}; "
            "score_anomaly now triggers auto-review and memory updates, not a human-wait pause."
        )
        save_memory(base)
        return base
    return base


def seed_memory_from_journal(memory: dict[str, Any]) -> dict[str, Any]:
    try:
        import prepare
        current_score_version = getattr(prepare, "SCORE_VERSION", "demo_v1")
    except Exception:
        current_score_version = "demo_v1"
    runs = [r for r in journal_runs(limit=200) if r.get("score_version", "demo_v1") == current_score_version]
    if not runs:
        memory["summary"] = f"No {current_score_version} service iterations have been memorized yet."
        memory["best_known"] = None
        memory["avoid"] = []
        memory["promising"] = []
        memory["recent"] = []
        memory["anomaly_reviews"] = []
        memory["anomaly_guidance"] = ""
        memory["score_version"] = current_score_version
        return memory
    accepted = [r for r in runs if r.get("decision") == "ACCEPTED"]
    best = max(accepted, key=lambda r: r.get("score") or float("-inf"), default=None)
    recent_items = []
    for run in runs[-12:]:
        recent_items.append({
            "ts": run.get("ts"),
            "service_iteration": None,
            "runner_iter": run.get("iter_id"),
            "decision": run.get("decision") or run.get("status"),
            "score": run.get("score"),
            "summary": f"Historical runner record: {run.get('decision') or run.get('status')}",
            "research_note": run.get("note_path"),
            "error": run.get("error"),
            "score_anomaly": run.get("score_anomaly"),
        })
    memory["recent"] = recent_items
    memory["anomaly_reviews"] = [
        {
            "runner_iter": r.get("iter_id"),
            "score": r.get("score"),
            "decision": r.get("decision") or r.get("status"),
            "score_anomaly": r.get("score_anomaly"),
        }
        for r in runs
        if r.get("score_anomaly")
    ][-8:]
    if best:
        memory["best_known"] = {
            "runner_iter": best.get("iter_id"),
            "score": best.get("score"),
            "summary": "Seeded from journal/runs.jsonl best accepted run.",
            "factor_library": best.get("factor_library"),
        }
    memory["promising"] = [
        f"runner#{r.get('iter_id')} ACCEPTED score={r.get('score')}: {r.get('factor_library') or r.get('note_path')}"
        for r in accepted[-8:]
    ]
    memory["avoid"] = [
        short_memory_line({
            "runner_iter": r.get("iter_id"),
            "decision": r.get("decision") or r.get("status"),
            "score": r.get("score"),
            "summary": r.get("error") or f"Rejected score={r.get('score')} below best={r.get('best_score')}",
        })
        for r in runs
        if r.get("decision") in {"REJECTED", "REVERTED"} or r.get("status") == "crash"
    ][-12:]
    if best:
        memory["summary"] = (
            f"Seeded from {current_score_version} journal: best accepted runner#{best.get('iter_id')} "
            f"score={best.get('score')}. Avoid repeating recent rejected/crashed variants."
        )
    else:
        memory["summary"] = f"Seeded from {current_score_version} journal, but no accepted run found yet."
    memory["score_version"] = current_score_version
    return memory


def save_memory(memory: dict[str, Any]) -> None:
    ensure_state()
    memory["updated_at"] = now_iso()
    MEMORY_PATH.write_text(json.dumps(json_safe(memory), indent=2, ensure_ascii=False), encoding="utf-8")


def reset_memory() -> dict[str, Any]:
    memory = default_memory()
    save_memory(memory)
    append_log("audit", "memory_reset", {})
    return memory


def memory_prompt_text(memory: dict[str, Any]) -> str:
    recent = memory.get("recent", [])[-8:]
    return json.dumps({
        "summary": memory.get("summary"),
        "best_known": memory.get("best_known"),
        "avoid": memory.get("avoid", [])[-12:],
        "promising": memory.get("promising", [])[-12:],
        "anomaly_guidance": memory.get("anomaly_guidance", ""),
        "anomaly_reviews": memory.get("anomaly_reviews", [])[-5:],
        "stage_guidance": memory.get("stage_guidance", ""),
        "recent": recent,
    }, ensure_ascii=False, indent=2)


def current_score_version() -> str:
    try:
        import prepare
        return getattr(prepare, "SCORE_VERSION", "demo_v1")
    except Exception:
        return "demo_v1"


def current_best_record() -> dict[str, Any] | None:
    best_path = ROOT / "journal" / "best.json"
    runs_path = ROOT / "journal" / "runs.jsonl"
    if not best_path.exists() or not runs_path.exists():
        return None
    try:
        best = json.loads(best_path.read_text(encoding="utf-8"))
        best_iter = best.get("iter_id")
    except Exception:
        best_iter = None
    records = []
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if best_iter is not None and rec.get("iter_id") == best_iter:
            return rec
        if rec.get("score_version") == current_score_version() and rec.get("decision") == "ACCEPTED":
            records.append(rec)
    return max(records, key=lambda r: r.get("score") or float("-inf"), default=None)


def detect_bottlenecks() -> dict[str, Any]:
    rec = current_best_record() or {}
    raw = ((rec.get("score_breakdown") or {}).get("raw") or {})
    weighted = ((rec.get("score_breakdown") or {}).get("weighted") or {})
    diag = rec.get("diag") or {}
    bottlenecks: list[dict[str, Any]] = []

    def add(key: str, label: str, severity: float, evidence: dict[str, Any], advice: str) -> None:
        if severity <= 0:
            return
        bottlenecks.append({
            "key": key,
            "label": label,
            "severity": round(float(severity), 4),
            "evidence": evidence,
            "advice": advice,
        })

    year_stability = raw.get("year_stability")
    positive_year_ratio = raw.get("positive_year_ratio")
    if isinstance(year_stability, int | float) or isinstance(positive_year_ratio, int | float):
        sev = max(0.0, -float(year_stability or 0.0)) + max(0.0, 0.67 - float(positive_year_ratio or 0.0))
        add(
            "year_stability_low",
            "年度稳定性偏弱",
            sev,
            {"year_stability": year_stability, "positive_year_ratio": positive_year_ratio, "weighted": weighted.get("year_stability_term")},
            "优先尝试降低年份间收益波动、减少单一年份依赖，保留低换手平滑带来的优势。",
        )

    turnover_penalty = raw.get("turnover_penalty")
    annual_turnover = raw.get("annual_turnover")
    if isinstance(turnover_penalty, int | float) or isinstance(annual_turnover, int | float):
        sev = max(0.0, float(turnover_penalty or 0.0)) + max(0.0, (float(annual_turnover or 0.0) - 40.0) / 40.0)
        add(
            "turnover_penalty_high",
            "换手惩罚偏高",
            sev,
            {"annual_turnover": annual_turnover, "turnover_penalty": turnover_penalty, "weighted": weighted.get("turnover_penalty_term")},
            "优先考虑最终信号平滑、权重稳定、删除高换手噪声因子；避免牺牲太多 IC。",
        )

    excess_efficiency = raw.get("excess_efficiency")
    return_efficiency = raw.get("return_efficiency")
    excess_sharpe = raw.get("excess_sharpe")
    sev = 0.0
    if isinstance(excess_efficiency, int | float):
        sev += max(0.0, 0.80 - float(excess_efficiency))
    if isinstance(return_efficiency, int | float):
        sev += max(0.0, 0.25 - float(return_efficiency))
    if isinstance(excess_sharpe, int | float):
        sev += max(0.0, 0.45 - float(excess_sharpe))
    add(
        "efficiency_low",
        "收益/回撤效率仍可提升",
        sev,
        {"excess_efficiency": excess_efficiency, "return_efficiency": return_efficiency, "excess_sharpe": excess_sharpe},
        "优先寻找提升收益质量而不增加回撤的改动，避免只追 IC。",
    )

    redundancy_penalty = raw.get("redundancy_penalty")
    max_factor_corr = raw.get("max_factor_corr")
    if isinstance(redundancy_penalty, int | float) or isinstance(max_factor_corr, int | float):
        sev = max(0.0, float(redundancy_penalty or 0.0)) + max(0.0, (float(max_factor_corr or 0.0) - 0.75) / 0.15)
        add(
            "redundancy_high",
            "因子同质化风险",
            sev,
            {"max_factor_corr": max_factor_corr, "redundancy_penalty": redundancy_penalty, "weighted": weighted.get("redundancy_penalty_term")},
            "避免直接追加相似因子；若探索新因子，先考虑残差化或删减替代。",
        )

    factor_count = raw.get("factor_count")
    alpha_lines = raw.get("alpha_lines")
    complexity_penalty = raw.get("complexity_penalty")
    code_complexity_penalty = raw.get("code_complexity_penalty")
    sev = max(0.0, float(complexity_penalty or 0.0)) + max(0.0, float(code_complexity_penalty or 0.0))
    add(
        "complexity_high",
        "复杂度偏高",
        sev,
        {"factor_count": factor_count, "alpha_lines": alpha_lines, "complexity_penalty": complexity_penalty, "code_complexity_penalty": code_complexity_penalty},
        "优先删减、合并或简化，而不是增加黑盒模型。",
    )

    # Phase 3 guardrail: recent bests came mainly from increasingly strong
    # smoothing. When turnover is already low and top-basket persistence is high,
    # further span/window tuning is more likely validation adaptation than true
    # alpha discovery.
    top_basket_jaccard = diag.get("top_basket_jaccard")
    score = rec.get("score")
    sev = 0.0
    if isinstance(score, int | float):
        sev += max(0.0, (float(score) - 7.0) / 2.0)
    if isinstance(annual_turnover, int | float):
        sev += max(0.0, (22.0 - float(annual_turnover)) / 22.0)
    if isinstance(top_basket_jaccard, int | float):
        sev += max(0.0, (float(top_basket_jaccard) - 0.90) * 4.0)
    add(
        "validation_overfit_risk",
        "平滑参数/验证集适配风险",
        sev,
        {
            "score": score,
            "annual_turnover": annual_turnover,
            "top_basket_jaccard": top_basket_jaccard,
            "recent_best_iter": rec.get("iter_id"),
        },
        "当前 best 已由多层平滑推高；优先做消融、删减、低相关新结构，避免继续只调 span/window/rolling median。",
    )

    bottlenecks.sort(key=lambda x: x["severity"], reverse=True)
    return {
        "score_version": rec.get("score_version") or current_score_version(),
        "best_iter": rec.get("iter_id"),
        "best_score": rec.get("score"),
        "top": bottlenecks[:3],
        "raw": raw,
        "weighted": weighted,
    }


def normalize_proposals(payload: dict[str, Any]) -> list[dict[str, Any]]:
    proposals = payload.get("proposals")
    if not isinstance(proposals, list):
        return []
    out = []
    for i, item in enumerate(proposals[:3], start=1):
        if not isinstance(item, dict):
            continue
        proposal = {
            "id": str(item.get("id") or f"p{i}"),
            "summary": str(item.get("summary") or item.get("title") or ""),
            "hypothesis": str(item.get("hypothesis") or ""),
            "change": str(item.get("change") or ""),
            "expected": str(item.get("expected") or ""),
            "op_type": str(item.get("op_type") or "other"),
            "family": str(item.get("family") or "unknown"),
            "primitive": str(item.get("primitive") or ""),
            "transform": str(item.get("transform") or ""),
            "target_bottleneck": str(item.get("target_bottleneck") or ""),
            "targets": item.get("targets") if isinstance(item.get("targets"), list) else [],
            "risk": str(item.get("risk") or ""),
        }
        if proposal["summary"] and proposal["hypothesis"] and proposal["change"]:
            out.append(proposal)
    return out


def score_proposal_for_gate(proposal: dict[str, Any], bottleneck: dict[str, Any], memory: dict[str, Any]) -> tuple[float, list[str]]:
    text = " ".join(str(proposal.get(k, "")) for k in (
        "summary", "hypothesis", "change", "expected", "risk",
        "family", "primitive", "transform", "target_bottleneck",
    )).lower()
    targets = " ".join(str(t).lower() for t in proposal.get("targets", []))
    combined = f"{text} {targets}"
    smoothing_terms = ("smooth", "smoothing", "ewm", "span", "rolling median", "window", "volatility_quantile", "平滑", "窗口")
    simplification_terms = ("ablation", "ablate", "stress", "simplify", "delete", "remove", "prune", "消融", "简化", "删除", "移除")
    is_smoothing_tweak = any(w in combined for w in smoothing_terms)
    is_smoothing_simplification = is_smoothing_tweak and any(w in combined for w in simplification_terms)
    exploration_families = {"price_volume_divergence", "liquidity_shock", "bar_structure", "orthogonal_residual"}
    low_corr_terms = (
        "low correlation", "low-corr", "orthogonal", "orthogonalize", "residual", "residualize",
        "different from", "distinct from", "replace", "substitute", "低相关", "正交", "残差", "差异", "替换",
    )
    turnover_guard_terms = ("low turnover", "turnover neutral", "no turnover increase", "不提高换手", "低换手", "换手不增加")
    score = 0.0
    reasons: list[str] = []
    for b in bottleneck.get("top", []):
        key = b.get("key")
        severity = float(b.get("severity") or 0.0)
        if key == "year_stability_low" and any(w in combined for w in ("year", "annual stability", "stability", "stable", "smooth", "jaccard", "年度", "稳定")):
            score += 4.0 + severity
            reasons.append("targets year stability")
        if key == "turnover_penalty_high" and any(w in combined for w in ("turnover", "smooth", "ewm", "holding", "换手", "平滑")):
            score += 3.5 + severity
            reasons.append("targets turnover")
        if key == "efficiency_low" and any(w in combined for w in ("efficiency", "drawdown", "sharpe", "return", "收益", "回撤", "效率")):
            score += 3.0 + severity
            reasons.append("targets efficiency")
        if key == "redundancy_high" and any(w in combined for w in ("orthogonal", "residual", "delete", "remove", "去相关", "残差", "删除")):
            score += 3.0 + severity
            reasons.append("targets redundancy")
        if key == "complexity_high" and any(w in combined for w in ("delete", "remove", "simplify", "删", "简化")):
            score += 2.5 + severity
            reasons.append("targets complexity")
        if key == "validation_overfit_risk":
            if any(w in combined for w in ("ablation", "ablate", "stress", "robust", "simplify", "delete", "remove", "prune", "orthogonal", "residual", "消融", "稳健", "简化", "删除", "正交", "残差")):
                score += 4.0 + severity
                reasons.append("targets validation overfit risk")
            if is_smoothing_tweak and not is_smoothing_simplification:
                score -= 3.0 + severity
                reasons.append("penalized smoothing-tweak under overfit risk")

    op_type = proposal.get("op_type")
    family = str(proposal.get("family") or "unknown")
    if family in ALLOWED_PROPOSAL_FAMILIES:
        score += 0.6
        reasons.append(f"known family: {family}")
    else:
        score -= 0.7
        reasons.append("unknown factor family")
    if family in {"path_quality", "low_turnover_state"}:
        score += 0.8
        reasons.append("preferred stability family")
    if family in {"price_volume_divergence", "liquidity_shock", "orthogonal_residual", "factor_pruning", "robustness_audit"}:
        score += 0.7
        reasons.append("diversifying family")
    if family == "bar_structure":
        score += 0.5
        reasons.append("underused structural family")
    if family in exploration_families:
        score += 1.4
        reasons.append("exploration bonus for low-correlation factor family")
        if any(w in combined for w in low_corr_terms):
            score += 1.0
            reasons.append("states low-correlation or orthogonalization rationale")
        if any(w in combined for w in turnover_guard_terms):
            score += 0.6
            reasons.append("states turnover guardrail")
        if any(w in combined for w in ("slow_vol_regime", "momentum_20", "price_range_position_10", "range_position", "vol regime", "慢波动", "动量")):
            score += 0.4
            reasons.append("contrasts with current core factors")
    if family == "signal_stability":
        score -= 0.8
        reasons.append("signal stability saturated in phase 3")
    if op_type == "add_factor":
        score -= 0.8
        reasons.append("add_factor risk")
        if family in exploration_families and any(w in combined for w in low_corr_terms):
            score += 0.9
            reasons.append("bounded add_factor exploration")
        elif family not in exploration_families:
            score -= 0.7
            reasons.append("unbounded add_factor outside exploration budget")
    if op_type in {"delete_factor", "robustness_audit"}:
        score += 0.8
    elif op_type in {"preprocess", "combine_method"}:
        score += 0.4
        if is_smoothing_tweak and not is_smoothing_simplification:
            score -= 2.0
            reasons.append("preprocess smoothing tweak is now low value")
    if any(w in combined for w in ("halflife", "half-life", "半衰期", "horizon", "label_kind", "窗口微调")):
        score -= 2.5
        reasons.append("likely local tweak")
    if is_smoothing_tweak and not is_smoothing_simplification:
        score -= 2.5
        reasons.append("repeated smoothing parameter tweak")
    recent_avoid = " ".join(str(x).lower() for x in memory.get("avoid", [])[-6:])
    if proposal.get("summary", "").lower()[:60] and proposal.get("summary", "").lower()[:60] in recent_avoid:
        score -= 3.0
        reasons.append("resembles recent avoid")
    if family != "unknown" and family in recent_avoid and family not in {"signal_stability", "path_quality"}:
        score -= 1.0
        reasons.append("family appears in recent avoid")
    if any(w in combined for w in ("lightgbm", "mlp", "torch", "ridgecv", "black box", "黑盒")):
        score -= 1.5
        reasons.append("complex model risk")
    return score, reasons


def select_proposal(proposals: list[dict[str, Any]], bottleneck: dict[str, Any], memory: dict[str, Any]) -> dict[str, Any]:
    if not proposals:
        raise RuntimeError("Proposal gate received no usable proposals.")
    scored = []
    for proposal in proposals:
        score, reasons = score_proposal_for_gate(proposal, bottleneck, memory)
        item = dict(proposal)
        item["gate_score"] = round(score, 4)
        item["gate_reasons"] = reasons
        scored.append(item)
    family_counts: dict[str, int] = {}
    for item in scored:
        family = str(item.get("family") or "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1
    for item in scored:
        family = str(item.get("family") or "unknown")
        if family != "unknown" and family_counts.get(family, 0) == 1:
            item["gate_score"] = round(float(item["gate_score"]) + 0.35, 4)
            item["gate_reasons"] = item.get("gate_reasons", []) + ["unique family in candidate set"]
        elif family != "unknown":
            item["gate_score"] = round(float(item["gate_score"]) - 0.35, 4)
            item["gate_reasons"] = item.get("gate_reasons", []) + ["duplicate family in candidate set"]
    if not any(str(item.get("family")) in {"price_volume_divergence", "liquidity_shock", "bar_structure", "orthogonal_residual"} for item in scored):
        for item in scored:
            item["gate_score"] = round(float(item["gate_score"]) - 1.0, 4)
            item["gate_reasons"] = item.get("gate_reasons", []) + ["candidate set lacks low-correlation exploration"]
    scored.sort(key=lambda x: x["gate_score"], reverse=True)
    candidates = [dict(item) for item in scored]
    selected = dict(candidates[0])
    selected["all_candidates"] = candidates
    return selected


def update_memory_after_iteration(
    iteration: int,
    proposal: dict[str, Any],
    rec: dict[str, Any] | None,
    anomaly_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    memory = load_memory()
    run = rec or {}
    decision = run.get("decision") or run.get("status") or "unknown"
    score = run.get("score")
    item = {
        "ts": now_iso(),
        "service_iteration": iteration,
        "runner_iter": run.get("iter_id"),
        "decision": decision,
        "score": score,
        "summary": proposal.get("summary"),
        "research_note": proposal.get("research_note"),
        "selected_proposal": proposal.get("selected_proposal"),
        "error": run.get("error"),
        "score_anomaly": run.get("score_anomaly"),
        "anomaly_review": anomaly_review,
    }
    recent = memory.get("recent", [])
    recent.append(item)
    memory["recent"] = recent[-20:]

    if run.get("decision") == "ACCEPTED":
        memory["best_known"] = {
            "runner_iter": run.get("iter_id"),
            "score": score,
            "summary": proposal.get("summary"),
            "factor_library": run.get("factor_library"),
        }
        memory["promising"] = (memory.get("promising", []) + [short_memory_line(item)])[-12:]
    elif run.get("status") == "crash" or run.get("decision") == "REVERTED":
        memory["avoid"] = (memory.get("avoid", []) + [short_memory_line(item)])[-12:]
    elif run.get("decision") == "REJECTED":
        memory["avoid"] = (memory.get("avoid", []) + [short_memory_line(item)])[-12:]
    if run.get("score_anomaly"):
        if anomaly_review:
            review_item = {
                "ts": now_iso(),
                "service_iteration": iteration,
                "runner_iter": run.get("iter_id"),
                "score": score,
                "decision": decision,
                "score_anomaly": run.get("score_anomaly"),
                "summary": anomaly_review.get("summary"),
                "root_cause": anomaly_review.get("root_cause"),
                "next_guidance": anomaly_review.get("next_guidance"),
                "component_updates": anomaly_review.get("component_updates"),
                "avoid_patterns": anomaly_review.get("avoid_patterns"),
            }
            reviews = memory.get("anomaly_reviews", [])
            reviews.append(review_item)
            memory["anomaly_reviews"] = reviews[-12:]
            guidance_parts = [
                str(anomaly_review.get("next_guidance") or "").strip(),
                "Avoid patterns: " + "; ".join(str(x) for x in anomaly_review.get("avoid_patterns", [])[:5])
                if isinstance(anomaly_review.get("avoid_patterns"), list) else "",
            ]
            memory["anomaly_guidance"] = " ".join(part for part in guidance_parts if part)[-1600:]
            for avoid in anomaly_review.get("avoid_patterns", []) if isinstance(anomaly_review.get("avoid_patterns"), list) else []:
                memory["avoid"] = (memory.get("avoid", []) + [f"score_anomaly runner#{run.get('iter_id')}: {avoid}"])[-12:]
            memory["summary"] = (
                f"Score anomaly on runner#{run.get('iter_id')} was auto-reviewed; "
                "continue 24/7 iteration using anomaly_guidance instead of pausing."
            )
        else:
            memory["summary"] = (
                f"Score anomaly on runner#{run.get('iter_id')} was recorded; continue 24/7 "
                "without pausing, but avoid repeating that rejected pattern."
            )
    elif memory.get("best_known"):
        best = memory["best_known"]
        memory["summary"] = f"Best remembered run #{best.get('runner_iter')} score={best.get('score')}; avoid repeating recent rejected/crashed variants."
    else:
        memory["summary"] = "No accepted service-generated improvement remembered yet; prioritize simple, single-hypothesis changes."
    save_memory(memory)
    append_log("research", "memory_updated", {
        "iteration": iteration,
        "decision": decision,
        "score": score,
        "recent_count": len(memory.get("recent", [])),
    })
    return memory


def short_memory_line(item: dict[str, Any]) -> str:
    text = item.get("summary") or item.get("research_note") or item.get("error") or "iteration"
    text = str(text).replace("\n", " ")
    if len(text) > 180:
        text = text[:179] + "…"
    return f"runner#{item.get('runner_iter')} {item.get('decision')} score={item.get('score')}: {text}"


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def run_cmd(args: list[str], timeout: int = 900) -> tuple[int, str]:
    append_log("action", "command_started", {"args": args})
    started = time.time()
    proc = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    output = proc.stdout[-20000:]
    append_log("action", "command_finished", {
        "args": args,
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - started, 2),
        "output_tail": output,
    })
    return proc.returncode, output


def read_text(path: str, max_chars: int = 24000) -> str:
    data = (ROOT / path).read_text(encoding="utf-8")
    if len(data) > max_chars:
        return data[:max_chars] + "\n\n...[truncated]..."
    return data


def latest_runner_status() -> str:
    py = str(PYTHON if PYTHON.exists() else "python")
    code, output = run_cmd([py, str(RUNNER), "status"], timeout=120)
    return output if code == 0 else f"status command failed:\n{output}"


def last_run_record() -> dict[str, Any] | None:
    path = ROOT / "journal" / "runs.jsonl"
    if not path.exists():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except Exception:
        return None


def score_series() -> list[dict[str, Any]]:
    path = ROOT / "journal" / "runs.jsonl"
    if not path.exists():
        return []
    try:
        import prepare
        current_score_version = getattr(prepare, "SCORE_VERSION", "demo_v1")
    except Exception:
        current_score_version = "demo_v1"
    rows = []
    best = float("-inf")
    active_score_version = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        score = rec.get("score")
        score_version = rec.get("score_version", "demo_v1")
        if score_version != current_score_version:
            continue
        if active_score_version is None:
            active_score_version = score_version
        elif score_version != active_score_version:
            active_score_version = score_version
            best = float("-inf")
        if isinstance(score, int | float):
            best = max(best, float(score))
            best_score = best
        else:
            best_score = None if best == float("-inf") else best
        rows.append({
            "iter_id": rec.get("iter_id"),
            "ts": rec.get("ts"),
            "score": score,
            "score_version": score_version,
            "best_score": best_score,
            "decision": rec.get("decision"),
            "status": rec.get("status"),
            "horizon": rec.get("horizon"),
            "label_kind": rec.get("label_kind"),
            "rank_ic_ir": rec.get("rank_ic_ir"),
            "rank_ic_mean": rec.get("rank_ic_mean"),
            "pearson_ic_mean": rec.get("pearson_ic_mean"),
            "sharpe": rec.get("sharpe"),
            "annual_return": rec.get("annual_return"),
            "max_drawdown": rec.get("max_drawdown"),
            "score_anomaly": rec.get("score_anomaly"),
        })
    return rows


def journal_runs(limit: int = 80) -> list[dict[str, Any]]:
    path = ROOT / "journal" / "runs.jsonl"
    if not path.exists():
        return []
    rows = []
    best = float("-inf")
    active_score_version = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        score = rec.get("score")
        score_version = rec.get("score_version", "demo_v1")
        if active_score_version is None:
            active_score_version = score_version
        elif score_version != active_score_version:
            active_score_version = score_version
            best = float("-inf")
        if isinstance(score, int | float):
            prev_best = None if best == float("-inf") else best
            is_new_best = score > best
            best = max(best, float(score))
            best_score = best
            delta_to_best = None if prev_best is None else float(score) - prev_best
        else:
            is_new_best = False
            best_score = None if best == float("-inf") else best
            delta_to_best = None
        score_raw = {}
        score_breakdown = rec.get("score_breakdown")
        if isinstance(score_breakdown, dict) and isinstance(score_breakdown.get("raw"), dict):
            score_raw = score_breakdown["raw"]
        slim = {
            "iter_id": rec.get("iter_id"),
            "ts": rec.get("ts"),
            "status": rec.get("status"),
            "decision": rec.get("decision"),
            "score": score,
            "score_version": score_version,
            "best_score": best_score,
            "delta_to_previous_best": delta_to_best,
            "is_new_best": is_new_best,
            "elapsed_sec": rec.get("elapsed_sec"),
            "horizon": rec.get("horizon"),
            "label_kind": rec.get("label_kind"),
            "rank_ic_ir": rec.get("rank_ic_ir"),
            "rank_ic_mean": rec.get("rank_ic_mean"),
            "pearson_ic_mean": rec.get("pearson_ic_mean"),
            "sharpe": rec.get("sharpe"),
            "annual_return": rec.get("annual_return"),
            "max_drawdown": rec.get("max_drawdown"),
            "annual_turnover": rec.get("annual_turnover"),
            "monotonicity": rec.get("monotonicity"),
            "excess_sharpe": rec.get("excess_sharpe"),
            "factor_count": rec.get("factor_count"),
            "alpha_lines": rec.get("alpha_lines"),
            "year_stability": score_raw.get("year_stability"),
            "positive_year_ratio": score_raw.get("positive_year_ratio"),
            "turnover_penalty": score_raw.get("turnover_penalty"),
            "complexity_penalty": score_raw.get("complexity_penalty"),
            "redundancy_penalty": score_raw.get("redundancy_penalty"),
            "error": rec.get("error"),
            "score_anomaly": rec.get("score_anomaly"),
            "note_path": rec.get("note_path"),
            "factor_library": rec.get("factor_library"),
        }
        rows.append(slim)
    return rows[-limit:]


def context_blocks() -> dict[str, Any]:
    cfg = load_config()
    memory_block = ""
    memory = load_memory()
    if cfg.get("memory_enabled", True):
        memory_block = f"""
Persistent service memory:
{memory_prompt_text(memory)}

Use this memory to avoid repeating failed ideas and to build on accepted or promising results.
"""
    return {
        "cfg": cfg,
        "memory": memory,
        "memory_block": memory_block,
        "program": read_text("program.md"),
        "evaluation": read_text("evaluation.md"),
        "factor_grammar": read_text("factor_grammar.md"),
        "alpha": read_text("alpha.py", max_chars=40000),
        "status": latest_runner_status(),
        "bottleneck": detect_bottlenecks(),
    }


def build_proposal_prompt(ctx: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are an autonomous quantitative research coding agent inside the AutoAlpha demo. "
        "First act as a research planner only. Do not write alpha.py yet. "
        "Return strict JSON only."
    )
    user = f"""
We need proposal candidates for the next AutoAlpha iteration.

Rules:
- Return a JSON object with exactly one key: proposals.
- proposals must contain exactly 3 candidate objects.
- Each object must contain: id, summary, hypothesis, change, expected, op_type, family, primitive, transform, target_bottleneck, targets, risk.
- Do not write alpha_py in this response.
- Each proposal must be a single-point experiment.
- The 3 proposals should use 3 different family values from factor_grammar.md whenever possible.
- Use factor_grammar.md as the allowed vocabulary for family/primitive/transform.
- Prefer candidates that address the bottleneck detector below.
- Avoid repeating recent rejected ideas in memory.
- Avoid tiny half-life/window/label tweaks unless the bottleneck clearly demands them.
- Phase 3 constraint: at most one proposal may be signal smoothing / preprocess span-window tuning.
- Phase 3 constraint: at least one proposal must be factor_pruning, orthogonal_residual, or robustness_audit.
- Phase 3 exploration budget: at least one proposal must explore a low-correlation factor family: price_volume_divergence, liquidity_shock, bar_structure, or orthogonal_residual.
- Phase 3 constraint: do not propose another plain EWM/rolling-median/span tweak unless it removes or simplifies an existing smoothing layer.
- For any add_factor proposal, explicitly state why it differs from slow_vol_regime_60, momentum_20, and price_range_position_10, and why turnover should not increase.
{ctx["memory_block"]}

Current runner status:
{ctx["status"]}

Bottleneck detector:
{json.dumps(ctx["bottleneck"], ensure_ascii=False, indent=2)}

program.md:
{ctx["program"]}

evaluation.md:
{ctx["evaluation"]}

factor_grammar.md:
{ctx["factor_grammar"]}

Current alpha.py summary only:
- Full code will be provided after proposal selection.
- Current best is in memory and runner status.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_prompt(selected_proposal: dict[str, Any], ctx: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are an autonomous quantitative research coding agent inside the AutoAlpha demo. "
        "Your only allowed code change is replacing alpha.py. Obey program.md and evaluation.md. "
        "Implement exactly the selected single research proposal. Return strict JSON only."
    )
    user = f"""
We selected one proposal through the local proposal gate. Implement only this proposal.

Selected proposal:
{json.dumps({k: v for k, v in selected_proposal.items() if k != "all_candidates"}, ensure_ascii=False, indent=2)}

Rules:
- Return a JSON object with keys: summary, research_note, alpha_py.
- alpha_py must be the complete replacement content for alpha.py.
- Do not modify prepare.py, runner.py, metrics.py, evaluation.md, program.md, data files, or service files.
- Do not include any source strings forbidden by program.md inside alpha_py.
- Keep the public contract: HORIZON, LABEL_KIND, ITER_NOTE, FACTORS, run(train_panel, val_panel).
- Use only train data to estimate any fitted parameters. Validation is for runner evaluation only.
- The implementation must match the selected proposal; do not switch to a different idea.
- The current phase is Phase 3 robustness: prefer low-correlation structure, pruning, orthogonal residuals, and simplification over more smoothing parameter tweaks.
- If the selected proposal is a robustness_audit or pruning idea, it is acceptable for ITER_NOTE expected score to be flat/slightly down if the change reduces validation overfit risk.
{ctx["memory_block"]}

Current runner status:
{ctx["status"]}

Bottleneck detector:
{json.dumps(ctx["bottleneck"], ensure_ascii=False, indent=2)}

program.md:
{ctx["program"]}

evaluation.md:
{ctx["evaluation"]}

factor_grammar.md:
{ctx["factor_grammar"]}

current alpha.py:
{ctx["alpha"]}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_score_anomaly_review_prompt(
    ctx: dict[str, Any],
    proposal: dict[str, Any],
    rec: dict[str, Any],
    runner_tail: str,
) -> list[dict[str, str]]:
    system = (
        "You are the AutoAlpha 24/7 research supervisor. "
        "A runner iteration triggered score_anomaly. Do not write code. "
        "Diagnose why the anomaly happened and produce operational guidance for the next loop. "
        "Return strict JSON only."
    )
    user = f"""
AutoAlpha must run unattended 24/7. A rejected run triggered score_anomaly, so we need an automatic review instead of waiting for a human.

Return a JSON object with these keys:
- summary: concise Chinese summary.
- root_cause: likely reason the score_anomaly fired.
- should_continue: true unless there is clear data leakage or infrastructure corruption.
- next_guidance: concrete guidance for the next proposal gate and implementation prompt.
- component_updates: list of suggested updates to service memory/proposal gate/evaluation notes/factor grammar. These are advisory memory updates, not direct file edits.
- avoid_patterns: list of patterns the next proposals should avoid.
- prefer_patterns: list of patterns the next proposals should prefer.

Current baseline / bottleneck:
{json.dumps(ctx.get("bottleneck"), ensure_ascii=False, indent=2)}

Persistent memory:
{memory_prompt_text(ctx.get("memory", {}))}

Selected proposal that led to the anomaly:
{json.dumps(proposal.get("selected_proposal") or proposal, ensure_ascii=False, indent=2)}

Runner record:
{json.dumps(rec, ensure_ascii=False, indent=2)}

Runner output tail:
{runner_tail[-4000:]}

Rules:
- If the anomaly is only "MDD improved but score is still much worse", do not stop the service.
- If score, Sharpe, annual_return and IC are all worse except one isolated risk metric, mark it as a low-quality anomaly and guide the next loop away from that pattern.
- If there is any hint of leakage, invalid data, or repeated infrastructure failure, set should_continue=false and explain.
- Keep guidance compatible with program.md and evaluation.md.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def fallback_score_anomaly_review(rec: dict[str, Any], proposal: dict[str, Any], error: str | None = None) -> dict[str, Any]:
    anomaly = rec.get("score_anomaly") or {}
    triggers = anomaly.get("triggers") or []
    return {
        "summary": "score_anomaly 已自动记录；本次不暂停服务，下一轮避开导致综合 score 明显下降的方向。",
        "root_cause": anomaly.get("message") or error or "Runner flagged an anomaly but no model review was available.",
        "should_continue": True,
        "next_guidance": (
            "Continue from current best. Treat isolated drawdown improvement as insufficient when Sharpe, return, "
            "IC or score degrade materially; prefer proposals that improve efficiency without collapsing predictive quality."
        ),
        "component_updates": [
            "memory.anomaly_reviews append fallback review",
            "proposal_gate penalize variants similar to the rejected anomaly proposal",
        ],
        "avoid_patterns": [
            "单独优化回撤但显著牺牲 Sharpe/annual_return/IC",
            "重复提交与上一个 score_anomaly proposal 相似的改动",
            *(str(x) for x in triggers[:3]),
        ],
        "prefer_patterns": [
            "保持 #0224 best 的低换手和高 Sharpe 特征",
            "优先提高 year_stability 与 excess_efficiency，不破坏 rank_ic_ir",
        ],
        "review_error": error,
        "selected_summary": proposal.get("summary"),
    }


def review_score_anomaly(
    cfg: dict[str, Any],
    ctx: dict[str, Any],
    proposal: dict[str, Any],
    rec: dict[str, Any],
    runner_tail: str,
) -> dict[str, Any]:
    if not cfg.get("auto_review_score_anomaly", True):
        return fallback_score_anomaly_review(rec, proposal, "auto_review_score_anomaly disabled")
    try:
        messages = build_score_anomaly_review_prompt(ctx, proposal, rec, runner_tail)
        append_log("research", "score_anomaly_review_context_built", {
            "runner_iter": rec.get("iter_id"),
            "score": rec.get("score"),
            "message_count": len(messages),
            "score_anomaly": rec.get("score_anomaly"),
        })
        review = call_openai_compatible(cfg, messages)
        if not isinstance(review, dict):
            raise RuntimeError("Review response was not a JSON object.")
        review.setdefault("should_continue", True)
        review.setdefault("summary", "score_anomaly auto-reviewed.")
        review.setdefault("avoid_patterns", [])
        review.setdefault("prefer_patterns", [])
        review.setdefault("component_updates", [])
        append_log("research", "score_anomaly_auto_reviewed", {
            "runner_iter": rec.get("iter_id"),
            "score": rec.get("score"),
            "review": review,
        })
        return review
    except Exception as exc:
        review = fallback_score_anomaly_review(rec, proposal, f"{type(exc).__name__}: {exc}")
        append_log("research", "score_anomaly_auto_review_failed", {
            "runner_iter": rec.get("iter_id"),
            "score": rec.get("score"),
            "error": review.get("review_error"),
            "fallback_review": review,
        })
        return review


def normalize_base_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if base.endswith("/v1"):
        return base + "/chat/completions"
    if base.endswith("/chat/completions"):
        return base
    return base + "/v1/chat/completions"


def normalize_models_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        base = base.removesuffix("/chat/completions")
    if base.endswith("/v1"):
        return base + "/models"
    return base + "/v1/models"


def post_json(url: str, body: dict[str, Any], api_key: str, timeout: int = 240) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API HTTP {exc.code}: {detail}") from exc


def call_openai_compatible(cfg: dict[str, Any], messages: list[dict[str, str]]) -> dict[str, Any]:
    if not cfg.get("base_url") or not cfg.get("api_key") or not cfg.get("model"):
        raise RuntimeError("API config is incomplete: base_url, api_key, and model are required.")
    body = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": cfg.get("temperature", 0.2),
        "response_format": {"type": "json_object"},
    }
    url = normalize_base_url(cfg["base_url"])
    try:
        data = post_json(url, body, cfg["api_key"], timeout=240)
    except RuntimeError as exc:
        # LM Studio and some compatible servers do not support OpenAI's
        # json_object mode. The prompt already requests strict JSON, so retry
        # with plain text instead of blocking local-model use.
        if "response_format" not in str(exc):
            raise
        fallback = dict(body)
        fallback.pop("response_format", None)
        data = post_json(url, fallback, cfg["api_key"], timeout=240)
    content = data["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model did not return JSON. Content starts with: {content[:500]}") from exc


def test_api_config(cfg: dict[str, Any]) -> dict[str, Any]:
    if not cfg.get("base_url") or not cfg.get("model"):
        raise RuntimeError("base_url and model are required for the connection test.")
    api_key = cfg.get("api_key") or ""
    models_req = urllib.request.Request(
        normalize_models_url(cfg["base_url"]),
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        method="GET",
    )
    with urllib.request.urlopen(models_req, timeout=10) as resp:
        models_data = json.loads(resp.read().decode("utf-8"))
    model_ids = [item.get("id") for item in models_data.get("data", []) if item.get("id")]
    chat_body = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": "You are testing an OpenAI-compatible local API."},
            {"role": "user", "content": "Reply with exactly: CONNECTED"},
        ],
        "temperature": 0,
        "max_tokens": 128,
    }
    chat_data = post_json(normalize_base_url(cfg["base_url"]), chat_body, api_key, timeout=60)
    message = chat_data.get("choices", [{}])[0].get("message", {})
    return {
        "ok": True,
        "base_url": cfg["base_url"],
        "model": cfg["model"],
        "models": model_ids,
        "chat_model": chat_data.get("model"),
        "content": message.get("content"),
        "reasoning_content": message.get("reasoning_content"),
        "usage": chat_data.get("usage"),
    }


def maybe_git_commit(cfg: dict[str, Any], rec: dict[str, Any] | None) -> None:
    if not cfg.get("auto_commit_accepted") or not rec or rec.get("decision") != "ACCEPTED":
        return
    code, status = run_cmd(["git", "status", "--short"], timeout=60)
    if code != 0:
        return
    dirty = [line for line in status.splitlines() if line and not line.startswith("?? service_state")]
    if not dirty:
        return
    run_id = rec.get("iter_id", "unknown")
    score = rec.get("score", 0.0)
    run_cmd(["git", "add", "alpha.py"], timeout=60)
    run_cmd(["git", "commit", "-m", f"AutoAlpha iteration {run_id}: score {score:.4f}"], timeout=120)


def one_iteration() -> bool:
    cfg = load_config()
    with STATE_LOCK:
        RUNTIME["iteration"] += 1
        iteration = RUNTIME["iteration"]
        RUNTIME["status"] = "calling_model"
        RUNTIME["last_started_at"] = now_iso()
        RUNTIME["last_error"] = None
    append_log("audit", "iteration_started", {"iteration": iteration})

    ctx = context_blocks()
    proposal_messages = build_proposal_prompt(ctx)
    append_log("research", "proposal_context_built", {
        "iteration": iteration,
        "message_count": len(proposal_messages),
        "bottleneck": ctx.get("bottleneck"),
    })
    proposal_payload = call_openai_compatible(cfg, proposal_messages)
    candidates = normalize_proposals(proposal_payload)
    selected = select_proposal(candidates, ctx["bottleneck"], ctx["memory"])
    append_log("research", "proposal_gate_selected", {
        "iteration": iteration,
        "selected": {k: v for k, v in selected.items() if k != "all_candidates"},
        "candidates": selected.get("all_candidates", []),
        "bottleneck": ctx.get("bottleneck"),
    })

    messages = build_prompt(selected, ctx)
    append_log("research", "implementation_context_built", {"iteration": iteration, "message_count": len(messages)})
    proposal = call_openai_compatible(cfg, messages)
    proposal["selected_proposal"] = {k: v for k, v in selected.items() if k != "all_candidates"}
    proposal["proposal_candidates"] = selected.get("all_candidates", [])
    alpha_py = proposal.get("alpha_py")
    if not isinstance(alpha_py, str) or "def run(" not in alpha_py or "ITER_NOTE" not in alpha_py:
        raise RuntimeError("Proposal missing a complete alpha_py with ITER_NOTE and run().")
    append_log("research", "proposal_received", {
        "iteration": iteration,
        "summary": proposal.get("summary"),
        "research_note": proposal.get("research_note"),
        "selected_proposal": proposal.get("selected_proposal"),
        "alpha_chars": len(alpha_py),
    })

    previous = ALPHA_PATH.read_text(encoding="utf-8")
    backup_path = STATE_DIR / f"alpha_before_service_iter_{iteration:04d}.py"
    backup_path.write_text(previous, encoding="utf-8")
    with STATE_LOCK:
        RUNTIME["status"] = "replacing_alpha"
    ALPHA_PATH.write_text(alpha_py, encoding="utf-8")
    append_log("action", "alpha_replaced", {"iteration": iteration, "backup": str(backup_path)})

    py = str(PYTHON if PYTHON.exists() else "python")
    with STATE_LOCK:
        RUNTIME["status"] = "evaluating"
    code, output = run_cmd([py, str(RUNNER), "once"], timeout=3600)
    rec = last_run_record()
    with STATE_LOCK:
        RUNTIME["status"] = "recording_delivery"
    append_log("delivery", "iteration_evaluated", {
        "iteration": iteration,
        "returncode": code,
        "runner_tail": output[-4000:],
        "run_record": rec,
    })
    anomaly_review = None
    if rec and rec.get("score_anomaly"):
        with STATE_LOCK:
            RUNTIME["status"] = "reviewing_score_anomaly"
        anomaly_review = review_score_anomaly(cfg, ctx, proposal, rec, output[-4000:])
        append_log("audit", "score_anomaly_reviewed_continue", {
            "iteration": iteration,
            "score": rec.get("score"),
            "decision": rec.get("decision"),
            "score_anomaly": rec.get("score_anomaly"),
            "should_continue": anomaly_review.get("should_continue", True),
            "summary": anomaly_review.get("summary"),
        })
    if cfg.get("memory_enabled", True):
        update_memory_after_iteration(iteration, proposal, rec, anomaly_review=anomaly_review)
    maybe_git_commit(cfg, rec)
    with STATE_LOCK:
        RUNTIME["status"] = "sleeping"
        RUNTIME["last_finished_at"] = now_iso()
    append_log("audit", "iteration_finished", {
        "iteration": iteration,
        "returncode": code,
        "decision": rec.get("decision") if rec else None,
        "score": rec.get("score") if rec else None,
    })
    return True


def loop_main() -> None:
    append_log("audit", "loop_started", {})
    while True:
        with STATE_LOCK:
            if RUNTIME["stop_requested"]:
                break
        cfg = load_config()
        if not cfg.get("base_url") or not cfg.get("api_key") or not cfg.get("model"):
            with STATE_LOCK:
                RUNTIME["status"] = "waiting_for_api_config"
            time.sleep(2)
            continue
        try:
            should_continue = one_iteration()
            if not should_continue:
                break
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            append_log("audit", "iteration_error", {
                "error": err,
                "traceback": traceback.format_exc()[-6000:],
            })
            with STATE_LOCK:
                RUNTIME["last_error"] = err
                RUNTIME["status"] = "error_sleeping"
        sleep_for = load_config().get("iteration_sleep_sec", 5)
        for _ in range(max(1, int(sleep_for))):
            with STATE_LOCK:
                if RUNTIME["stop_requested"]:
                    break
            time.sleep(1)
    with STATE_LOCK:
        final_status = RUNTIME["status"]
        RUNTIME["running"] = False
        RUNTIME["stop_requested"] = False
        RUNTIME["status"] = "idle"
    append_log("audit", "loop_stopped", {})


def start_loop() -> None:
    global LOOP_THREAD
    with STATE_LOCK:
        if RUNTIME["running"]:
            return
        RUNTIME["running"] = True
        RUNTIME["stop_requested"] = False
        RUNTIME["status"] = "starting"
    LOOP_THREAD = threading.Thread(target=loop_main, daemon=True)
    LOOP_THREAD.start()


def stop_loop() -> None:
    with STATE_LOCK:
        RUNTIME["stop_requested"] = True
        RUNTIME["status"] = "stopping"
    append_log("audit", "stop_requested", {})


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AutoAlpha Service</title>
  <style>
    :root { color-scheme: light; --bg:#f6f7f9; --panel:#fff; --ink:#111827; --muted:#667085; --line:#d7dce3; --ok:#087443; --bad:#b42318; --accent:#1d4ed8; }
    * { box-sizing: border-box; }
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--ink); }
    header { padding:18px 24px; border-bottom:1px solid var(--line); background:#fff; display:flex; justify-content:space-between; gap:16px; align-items:center; }
    h1 { margin:0; font-size:20px; }
    main { padding:20px 24px; display:grid; grid-template-columns:360px 1fr; gap:20px; align-items:start; }
    section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; }
    label { display:block; font-size:12px; color:var(--muted); margin:12px 0 6px; }
    input, select, textarea { width:100%; border:1px solid var(--line); border-radius:6px; padding:9px 10px; font:inherit; background:#fff; }
    textarea { min-height:76px; resize:vertical; }
    button { border:1px solid var(--line); border-radius:6px; padding:9px 12px; background:#fff; cursor:pointer; font-weight:600; }
    button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
    button.danger { background:#fff5f5; color:var(--bad); border-color:#f3b8b0; }
    .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
    .status { font-size:13px; color:var(--muted); }
    .pill { display:inline-block; padding:4px 8px; border-radius:999px; background:#eef2ff; color:#3730a3; font-size:12px; font-weight:700; }
    .tabs { display:flex; gap:8px; margin-bottom:12px; }
    .tabs button.active { background:#111827; color:#fff; border-color:#111827; }
    pre { margin:0; white-space:pre-wrap; word-break:break-word; font-size:12px; line-height:1.45; }
    .logCards { height:620px; overflow:auto; border:1px solid var(--line); border-radius:8px; padding:12px; background:#f8fafc; display:grid; align-content:start; gap:10px; }
    .logToolbar { display:flex; justify-content:space-between; gap:10px; align-items:center; margin-bottom:10px; flex-wrap:wrap; }
    .logCard { background:#fff; border:1px solid var(--line); border-left:5px solid #94a3b8; border-radius:8px; padding:11px 12px; display:grid; gap:8px; }
    .logCard.ok { border-left-color:#087443; }
    .logCard.warn { border-left-color:#b54708; }
    .logCard.bad { border-left-color:#b42318; }
    .logCard.info { border-left-color:#1d4ed8; }
    .logTop { display:flex; justify-content:space-between; gap:10px; align-items:flex-start; }
    .logTitle { font-weight:700; line-height:1.3; }
    .logTime { color:var(--muted); font-size:11px; white-space:nowrap; }
    .logSummary { color:#344054; font-size:13px; line-height:1.45; }
    .chips { display:flex; gap:6px; flex-wrap:wrap; }
    .chip { border:1px solid var(--line); border-radius:999px; padding:3px 7px; background:#f8fafc; color:#475467; font-size:11px; }
    .logCard details { border-top:1px solid #eef2f7; padding-top:7px; }
    .logCard summary { cursor:pointer; color:var(--muted); font-size:12px; }
    .logCard pre { margin-top:8px; max-height:260px; overflow:auto; background:#0b1020; color:#dbeafe; border-radius:6px; padding:10px; }
    .muted { color:var(--muted); }
    .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .right { display:grid; gap:20px; }
    .chartHead { display:flex; justify-content:space-between; gap:12px; align-items:flex-start; margin-bottom:10px; }
    .metricRow { display:flex; gap:8px; flex-wrap:wrap; }
    .metric { border:1px solid var(--line); border-radius:6px; padding:8px 10px; min-width:116px; background:#fafafa; }
    .metric b { display:block; font-size:17px; }
    .metric span { color:var(--muted); font-size:11px; }
    #scoreChart { width:100%; height:280px; border:1px solid var(--line); border-radius:8px; background:#fbfcfe; display:block; }
    .legend { display:flex; gap:14px; flex-wrap:wrap; font-size:12px; color:var(--muted); margin-top:8px; }
    .dot { width:9px; height:9px; border-radius:50%; display:inline-block; margin-right:5px; }
    .flowWrap { display:grid; gap:12px; }
    .flow { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:10px; position:relative; }
    .flowStep { border:1px solid var(--line); border-radius:8px; padding:11px 12px; min-height:74px; background:#fafafa; position:relative; transition:all .2s ease; }
    .flowStep::after { content:""; position:absolute; top:50%; right:-10px; width:10px; height:2px; background:#cbd5e1; }
    .flowStep:nth-child(4)::after, .flowStep:nth-child(8)::after { display:none; }
    .flowStep small { display:block; color:var(--muted); font-size:11px; margin-bottom:5px; }
    .flowStep b { display:block; font-size:14px; margin-bottom:4px; }
    .flowStep span { color:var(--muted); font-size:12px; line-height:1.35; }
    .flowStep.done { border-color:#a7f3d0; background:#ecfdf3; }
    .flowStep.active { border-color:#1d4ed8; background:#eff6ff; box-shadow:0 0 0 3px rgba(29,78,216,.12); }
    .flowStep.wait { border-color:#fbbf24; background:#fffbeb; }
    .flowStep.error { border-color:#fca5a5; background:#fff1f2; }
    .flowStatus { color:var(--muted); font-size:13px; }
    .memoryBox { border:1px solid var(--line); border-radius:8px; padding:10px; background:#fbfcfe; display:grid; gap:8px; }
    .memoryBox ul { margin:0; padding-left:18px; color:#344054; font-size:12px; line-height:1.45; }
    .memoryBox p { margin:0; color:#344054; font-size:13px; line-height:1.45; }
    @media (max-width: 900px) { main { grid-template-columns:1fr; } .log { height:480px; } }
    @media (max-width: 1100px) { .flow { grid-template-columns:repeat(2, minmax(0, 1fr)); } .flowStep:nth-child(even)::after { display:none; } .flowStep:nth-child(4)::after { display:none; } }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>AutoAlpha Continuous Research Service</h1>
      <div class="status" id="statusText">loading...</div>
    </div>
    <div class="row">
      <span class="pill" id="runPill">idle</span>
      <button class="primary" onclick="startLoop()">启动持续迭代</button>
      <button class="danger" onclick="stopLoop()">停止</button>
    </div>
  </header>
  <main>
    <section>
      <h2>API 配置</h2>
      <label>Base URL</label>
      <input id="base_url" placeholder="https://api.example.com/v1" />
      <label>API Key</label>
      <input id="api_key" type="password" placeholder="保存后会隐藏" />
      <label>Model</label>
      <input id="model" placeholder="gpt-4.1 / qwen... / compatible-model" />
      <div class="grid2">
        <div>
          <label>Temperature</label>
          <input id="temperature" type="number" step="0.1" min="0" max="2" />
        </div>
        <div>
          <label>迭代间隔秒</label>
          <input id="iteration_sleep_sec" type="number" min="0" />
        </div>
      </div>
      <label class="row"><input id="auto_commit_accepted" type="checkbox" style="width:auto" /> accepted 后自动 git commit alpha.py</label>
      <label class="row"><input id="memory_enabled" type="checkbox" style="width:auto" /> 启用连续上下文记忆</label>
      <div class="row" style="margin-top:14px">
        <button class="primary" onclick="saveConfig()">保存配置</button>
        <button onclick="fillLmStudio()">填入 LM Studio</button>
        <button onclick="testConnection()">测试连接</button>
        <button onclick="refreshAll()">刷新</button>
      </div>
      <pre id="testResult" class="muted" style="margin-top:12px"></pre>
      <hr style="border:0;border-top:1px solid var(--line);margin:18px 0" />
      <h2>连续记忆</h2>
      <div class="memoryBox">
        <p id="memorySummary">loading...</p>
        <div class="chips" id="memoryChips"></div>
        <details>
          <summary>查看记忆详情</summary>
          <pre id="memoryDetails"></pre>
        </details>
        <button onclick="resetMemory()">清空记忆</button>
      </div>
      <hr style="border:0;border-top:1px solid var(--line);margin:18px 0" />
      <h2>人工追加日志</h2>
      <label>日志类型</label>
      <select id="manual_kind"><option>audit</option><option>action</option><option>research</option><option>delivery</option></select>
      <label>内容</label>
      <textarea id="manual_text" placeholder="写入一条人工备注"></textarea>
      <button onclick="appendManualLog()">追加日志</button>
      <p class="muted">后台循环不会自己停止；只有点击停止或进程退出才会停。</p>
    </section>
    <div class="right">
      <section>
        <div class="chartHead">
          <div>
            <h2 style="margin:0 0 4px">Score 实时曲线</h2>
            <div class="status" id="chartSubtitle">等待迭代数据</div>
          </div>
          <div class="metricRow">
            <div class="metric"><span>latest</span><b id="latestScore">--</b></div>
            <div class="metric"><span>best</span><b id="bestScore">--</b></div>
            <div class="metric"><span>iterations</span><b id="iterCount">0</b></div>
          </div>
        </div>
        <svg id="scoreChart" viewBox="0 0 900 280" role="img" aria-label="score by iteration"></svg>
        <div class="legend">
          <span><i class="dot" style="background:#087443"></i>ACCEPTED</span>
          <span><i class="dot" style="background:#b54708"></i>REJECTED</span>
          <span><i class="dot" style="background:#b42318"></i>CRASH/ERROR</span>
          <span><i class="dot" style="background:#1d4ed8"></i>best score</span>
          <span><i class="dot" style="background:#7c3aed"></i>score anomaly</span>
        </div>
      </section>
      <section>
        <div class="chartHead">
          <div>
            <h2 style="margin:0 0 4px">研究流程图</h2>
            <div class="flowStatus" id="flowStatus">等待状态同步</div>
          </div>
          <span class="pill" id="flowPill">idle</span>
        </div>
        <div class="flowWrap">
          <div class="flow">
            <div class="flowStep" data-step="config"><small>01</small><b>API 配置</b><span>LM Studio / 兼容 API 就绪</span></div>
            <div class="flowStep" data-step="context"><small>02</small><b>读取上下文</b><span>program、evaluation、alpha、best</span></div>
            <div class="flowStep" data-step="model"><small>03</small><b>生成候选</b><span>模型产出完整 alpha.py</span></div>
            <div class="flowStep" data-step="replace"><small>04</small><b>替换文件</b><span>备份旧版并写入候选</span></div>
            <div class="flowStep" data-step="evaluate"><small>05</small><b>运行回测</b><span>runner.py once 评分</span></div>
            <div class="flowStep" data-step="decision"><small>06</small><b>接受/回滚</b><span>ACCEPTED 留存，否则回滚</span></div>
            <div class="flowStep" data-step="deliver"><small>07</small><b>交付记录</b><span>score、日志、图表刷新</span></div>
            <div class="flowStep" data-step="loop"><small>08</small><b>下一轮/暂停</b><span>继续迭代或等待人工复核</span></div>
          </div>
        </div>
      </section>
      <section>
        <div class="chartHead">
          <div>
            <h2 style="margin:0 0 4px">Runner 实验记录</h2>
            <div class="status" id="runsHint">实时读取 journal/runs.jsonl</div>
          </div>
          <button onclick="refreshRuns()">刷新实验记录</button>
        </div>
        <div class="logCards" id="runsView" style="height:420px"></div>
      </section>
      <section>
        <div class="tabs">
          <button id="tab_audit" onclick="setLog('audit')">审计日志</button>
          <button id="tab_action" onclick="setLog('action')">行动日志</button>
          <button id="tab_research" onclick="setLog('research')">研究日志</button>
          <button id="tab_delivery" class="active" onclick="setLog('delivery')">交付日志</button>
        </div>
        <div class="logToolbar">
          <div class="status" id="logHint">卡片展示最近日志</div>
          <label class="row" style="margin:0"><input id="showHttpLogs" type="checkbox" style="width:auto" onchange="refreshLog()" /> 显示 HTTP 轮询</label>
        </div>
        <div class="logCards" id="logView"></div>
      </section>
    </div>
  </main>
  <script>
    let currentLog = 'delivery';
    async function api(path, options={}) {
      const res = await fetch(path, {headers:{'Content-Type':'application/json'}, ...options});
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }
    function $(id){ return document.getElementById(id); }
    async function refreshStatus() {
      const data = await api('/api/status');
      $('runPill').textContent = data.runtime.status;
      $('statusText').textContent = `running=${data.runtime.running} iteration=${data.runtime.iteration} last_error=${data.runtime.last_error || 'none'}`;
      updateFlow(data.runtime);
    }
    function updateFlow(runtime) {
      const status = runtime.status || 'idle';
      const activeByStatus = {
        idle: 'config',
        starting: 'context',
        waiting_for_api_config: 'config',
        calling_model: 'model',
        replacing_alpha: 'replace',
        evaluating: 'evaluate',
        recording_delivery: 'deliver',
        reviewing_score_anomaly: 'deliver',
        sleeping: 'loop',
        stopping: 'loop',
        error_sleeping: 'loop',
        waiting_for_human_score_anomaly: 'loop'
      };
      const order = ['config','context','model','replace','evaluate','decision','deliver','loop'];
      const active = activeByStatus[status] || 'loop';
      const activeIdx = order.indexOf(active);
      document.querySelectorAll('.flowStep').forEach(node => {
        const step = node.dataset.step;
        const idx = order.indexOf(step);
        node.classList.remove('active', 'done', 'wait', 'error');
        if (idx < activeIdx && runtime.running) node.classList.add('done');
        if (step === active) node.classList.add('active');
        if (status === 'waiting_for_api_config' && step === 'config') node.classList.add('wait');
        if (status === 'waiting_for_human_score_anomaly' && step === 'loop') node.classList.add('wait');
        if (status === 'reviewing_score_anomaly' && step === 'deliver') node.classList.add('wait');
        if (status === 'error_sleeping' && step === 'loop') node.classList.add('error');
      });
      const labels = {
        idle: '服务已启动，等待你点击“启动持续迭代”。',
        starting: '正在启动后台循环。',
        waiting_for_api_config: '等待填写并验证兼容 API 配置。',
        calling_model: '正在把研究上下文发给模型生成候选 alpha.py。',
        replacing_alpha: '正在备份旧版并写入模型生成的 alpha.py。',
        evaluating: '正在运行 runner.py once，等待 score 和回测指标。',
        recording_delivery: '正在写入交付日志并刷新 score 曲线。',
        reviewing_score_anomaly: '发现 score anomaly，正在自动调用模型复盘并更新连续记忆。',
        sleeping: '本轮已交付，等待下一轮迭代间隔。',
        stopping: '正在停止后台循环。',
        error_sleeping: '上一轮出错，服务短暂停顿后可复查日志。',
        waiting_for_human_score_anomaly: '旧状态：发现 score anomaly 后等待人工复核。当前版本会自动复盘后继续。'
      };
      $('flowStatus').textContent = labels[status] || `当前状态：${status}`;
      $('flowPill').textContent = status;
    }
    async function refreshConfig() {
      const cfg = await api('/api/config');
      for (const k of ['base_url','api_key','model','temperature','iteration_sleep_sec']) $(k).value = cfg[k] || '';
      $('auto_commit_accepted').checked = !!cfg.auto_commit_accepted;
      $('memory_enabled').checked = cfg.memory_enabled !== false;
    }
    async function saveConfig() {
      await api('/api/config', {method:'POST', body:JSON.stringify({
        base_url:$('base_url').value, api_key:$('api_key').value, model:$('model').value,
        temperature:parseFloat($('temperature').value || '0.2'),
        iteration_sleep_sec:parseInt($('iteration_sleep_sec').value || '5', 10),
        auto_commit_accepted:$('auto_commit_accepted').checked,
        memory_enabled:$('memory_enabled').checked
      })});
      await refreshAll();
    }
    function fillLmStudio() {
      $('base_url').value = 'http://127.0.0.1:1234/v1';
      $('api_key').value = 'lm-studio';
      if (!$('model').value) $('model').value = 'google/gemma-4-12b-qat';
    }
    async function testConnection() {
      $('testResult').textContent = 'testing...';
      try {
        const data = await api('/api/test-api', {method:'POST', body:JSON.stringify({
          base_url:$('base_url').value, api_key:$('api_key').value, model:$('model').value,
          temperature:parseFloat($('temperature').value || '0.2')
        })});
        $('testResult').textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        $('testResult').textContent = String(err);
      }
    }
    async function startLoop(){ await api('/api/start', {method:'POST', body:'{}'}); await refreshAll(); }
    async function stopLoop(){ await api('/api/stop', {method:'POST', body:'{}'}); await refreshAll(); }
    function setLog(kind){
      currentLog = kind;
      for (const k of ['audit','action','research','delivery']) $('tab_'+k).classList.toggle('active', k===kind);
      refreshLog();
    }
    async function refreshLog() {
      const rows = await api('/api/logs?kind=' + encodeURIComponent(currentLog) + '&limit=200');
      renderLogCards(rows);
    }
    function shortText(text, max=260) {
      const s = String(text ?? '').replace(/\s+/g, ' ').trim();
      return s.length > max ? s.slice(0, max - 1) + '…' : s;
    }
    function pct(v) {
      return Number.isFinite(Number(v)) ? (Number(v) * 100).toFixed(2) + '%' : '--';
    }
    function logCardClass(rec) {
      const p = rec.payload || {};
      const run = p.run_record || {};
      if (rec.event?.includes('error') || run.status === 'crash' || run.decision === 'REVERTED' || p.returncode > 0) return 'bad';
      if (run.decision === 'REJECTED' || rec.event?.includes('paused') || rec.event?.includes('stop')) return 'warn';
      if (run.decision === 'ACCEPTED' || rec.event?.includes('saved') || rec.event?.includes('test')) return 'ok';
      return 'info';
    }
    function runCardClass(run) {
      if (run.score_anomaly) return 'warn';
      if (run.status === 'crash' || run.decision === 'REVERTED') return 'bad';
      if (run.decision === 'ACCEPTED') return 'ok';
      if (run.decision === 'REJECTED') return 'warn';
      return 'info';
    }
    function runSummary(run) {
      if (run.error) return shortText(run.error, 360);
      const gap = Number(run.delta_to_previous_best);
      const gapText = Number.isFinite(gap) ? `gap ${gap >= 0 ? '+' : ''}${fmt(gap)}` : 'first baseline';
      return `score ${fmt(Number(run.score))} · ${gapText} · rank_ic_ir ${fmt(Number(run.rank_ic_ir), 3)} · sharpe ${fmt(Number(run.sharpe), 3)}`;
    }
    function renderRunCards(rows) {
      const view = $('runsView');
      view.innerHTML = '';
      const reversed = rows.slice().reverse();
      $('runsHint').textContent = `journal/runs.jsonl · ${rows.length} runs · 最新在上`;
      if (!reversed.length) {
        appendText(view, 'div', 'status', '暂无 runner 实验记录。');
        return;
      }
      reversed.forEach(run => {
        const card = document.createElement('article');
        card.className = `logCard ${runCardClass(run)}`;
        const top = document.createElement('div');
        top.className = 'logTop';
        const label = run.is_new_best ? 'NEW BEST' : (run.decision || run.status || '--');
        appendText(top, 'div', 'logTitle', `Run #${String(run.iter_id).padStart(4, '0')} · ${label}`);
        appendText(top, 'div', 'logTime', run.ts || '');
        card.appendChild(top);
        appendText(card, 'div', 'logSummary', runSummary(run));
        const chips = document.createElement('div');
        chips.className = 'chips';
        [
          `score ${fmt(Number(run.score))}`,
          `best ${fmt(Number(run.best_score))}`,
          run.score_version || '--',
          `H ${run.horizon ?? '--'}`,
          `turnover ${fmt(Number(run.annual_turnover), 1)}`,
          `mdd ${pct(run.max_drawdown)}`,
          `ret ${pct(run.annual_return)}`,
          Number.isFinite(Number(run.year_stability)) ? `year ${fmt(Number(run.year_stability), 2)}` : null,
          Number.isFinite(Number(run.positive_year_ratio)) ? `posY ${pct(run.positive_year_ratio)}` : null,
          Number.isFinite(Number(run.factor_count)) ? `factors ${run.factor_count}` : null,
          Number.isFinite(Number(run.complexity_penalty)) && Number(run.complexity_penalty) > 0 ? `complex ${fmt(Number(run.complexity_penalty), 2)}` : null,
          Number.isFinite(Number(run.redundancy_penalty)) && Number(run.redundancy_penalty) > 0 ? `redundant ${fmt(Number(run.redundancy_penalty), 2)}` : null,
          run.score_anomaly ? 'score anomaly' : null,
        ].filter(Boolean).forEach(c => appendText(chips, 'span', 'chip', c));
        card.appendChild(chips);
        if (run.note_path || run.factor_library) {
          appendText(card, 'div', 'logSummary', [run.note_path, run.factor_library].filter(Boolean).join(' · '));
        }
        const details = document.createElement('details');
        const summary = document.createElement('summary');
        summary.textContent = '实验详情';
        const pre = document.createElement('pre');
        pre.textContent = JSON.stringify(run, null, 2);
        details.appendChild(summary);
        details.appendChild(pre);
        card.appendChild(details);
        view.appendChild(card);
      });
    }
    async function refreshMemory() {
      const memory = await api('/api/memory');
      $('memorySummary').textContent = memory.summary || 'No memory summary yet.';
      $('memoryDetails').textContent = JSON.stringify(memory, null, 2);
      const chips = $('memoryChips');
      chips.innerHTML = '';
      [
        memory.score_version || 'score_version --',
        `recent ${memory.recent?.length || 0}`,
        `avoid ${memory.avoid?.length || 0}`,
        `promising ${memory.promising?.length || 0}`,
        memory.best_known ? `best #${memory.best_known.runner_iter}` : 'no remembered best',
      ].forEach(c => appendText(chips, 'span', 'chip', c));
    }
    async function resetMemory() {
      await api('/api/memory/reset', {method:'POST', body:'{}'});
      await refreshMemory();
    }
    function logSummary(rec) {
      const p = rec.payload || {};
      const run = p.run_record || {};
      if (rec.kind === 'delivery' && run.iter_id) {
        const decision = run.decision || run.status;
        return {
          title: `Run #${String(run.iter_id).padStart(4, '0')} · ${decision}`,
          summary: run.error ? shortText(run.error, 300) : `score ${fmt(Number(run.score))}, rank_ic_ir ${fmt(Number(run.rank_ic_ir), 3)}, sharpe ${fmt(Number(run.sharpe), 3)}`,
          chips: [`service iter ${p.iteration}`, `score ${fmt(Number(run.score))}`, `turnover ${fmt(Number(run.annual_turnover), 1)}`, `ret ${pct(run.annual_return)}`],
        };
      }
      if (rec.kind === 'research' && rec.event === 'proposal_received') {
        return {
          title: `研究提案 · service iter ${p.iteration}`,
          summary: shortText(p.summary || p.research_note || '模型返回候选 alpha.py', 320),
          chips: [`alpha ${p.alpha_chars || 0} chars`, p.selected_proposal?.op_type || rec.event],
        };
      }
      if (rec.kind === 'research' && rec.event === 'proposal_context_built') {
        const top = p.bottleneck?.top?.map(x => x.label).join(' / ') || 'no bottleneck';
        return { title: `Proposal gate · 候选请求`, summary: `先请求 3 个候选方向。当前瓶颈：${top}`, chips: [`iter ${p.iteration}`, `messages ${p.message_count}`] };
      }
      if (rec.kind === 'research' && rec.event === 'proposal_gate_selected') {
        const selected = p.selected || {};
        return { title: `Proposal gate · 已选择`, summary: shortText(selected.summary || selected.hypothesis || '已选择一个候选进入实现阶段。', 320), chips: [`iter ${p.iteration}`, `gate ${fmt(Number(selected.gate_score), 2)}`, selected.family || '--', selected.op_type || '--'] };
      }
      if (rec.kind === 'research' && rec.event === 'implementation_context_built') {
        return { title: `实现上下文已构建`, summary: `已把 gate 选中的候选和完整 alpha.py 发给模型实现。`, chips: [`iter ${p.iteration}`, `messages ${p.message_count}`] };
      }
      if (rec.kind === 'research' && rec.event === 'context_built') {
        return { title: `上下文已构建 · service iter ${p.iteration}`, summary: `已收集 program、evaluation、当前 alpha 与 runner status。`, chips: [`messages ${p.message_count}`] };
      }
      if (rec.kind === 'action' && rec.event === 'command_finished') {
        const args = Array.isArray(p.args) ? p.args.slice(-2).join(' ') : 'command';
        const tail = p.output_tail || '';
        const match = tail.match(/\[(ACCEPTED|REJECTED|CRASH)\][^\n]*/);
        return {
          title: `命令完成 · ${args}`,
          summary: match ? match[0] : shortText(tail, 260),
          chips: [`rc ${p.returncode}`, `${fmt(Number(p.elapsed_sec), 2)}s`],
        };
      }
      if (rec.kind === 'action' && rec.event === 'command_started') {
        const args = Array.isArray(p.args) ? p.args.slice(-2).join(' ') : 'command';
        return { title: `命令开始 · ${args}`, summary: `后台已开始执行命令。`, chips: [rec.event] };
      }
      if (rec.kind === 'action' && rec.event === 'alpha_replaced') {
        return { title: `alpha.py 已替换`, summary: `已备份旧版本并写入模型候选。`, chips: [`service iter ${p.iteration}`, 'backup saved'] };
      }
      if (rec.kind === 'audit' && rec.event === 'api_connection_test') {
        return { title: `API 连接测试成功`, summary: `模型 ${p.chat_model || p.model} 已响应；可用模型 ${Array.isArray(p.models) ? p.models.length : 0} 个。`, chips: [p.base_url, p.model] };
      }
      if (rec.kind === 'audit' && rec.event === 'config_saved') {
        return { title: `配置已保存`, summary: `模型 ${p.model || '--'}，API ${p.base_url || '--'}。`, chips: [p.has_api_key ? 'has key' : 'no key', `sleep ${p.iteration_sleep_sec}s`] };
      }
      if (rec.kind === 'audit' && rec.event === 'http') {
        return { title: `HTTP 请求`, summary: p.message || '', chips: [p.client || 'local'] };
      }
      if (rec.kind === 'audit' && rec.event === 'paused_for_score_anomaly') {
        return { title: `因 score anomaly 暂停`, summary: p.score_anomaly?.message || 'score 与底层指标背离，等待人工复核。', chips: [`score ${fmt(Number(p.score))}`, p.decision || '--'] };
      }
      if (rec.kind === 'audit' && rec.event === 'score_anomaly_reviewed_continue') {
        return { title: `score anomaly 已自动复盘`, summary: p.summary || p.score_anomaly?.message || '已写入记忆并继续下一轮。', chips: [`score ${fmt(Number(p.score))}`, p.decision || '--', p.should_continue === false ? 'review warned' : 'continue'] };
      }
      if (rec.kind === 'research' && rec.event === 'score_anomaly_auto_reviewed') {
        return { title: `异常复盘完成`, summary: p.review?.summary || p.review?.root_cause || '', chips: [`runner #${p.runner_iter}`, `score ${fmt(Number(p.score))}`] };
      }
      if (rec.kind === 'research' && rec.event === 'score_anomaly_auto_review_failed') {
        return { title: `异常复盘使用 fallback`, summary: p.error || '', chips: [`runner #${p.runner_iter}`, 'continue'] };
      }
      return {
        title: rec.event || rec.kind,
        summary: shortText(JSON.stringify(p, null, 2), 320),
        chips: [rec.kind],
      };
    }
    function appendText(parent, tag, className, text) {
      const el = document.createElement(tag);
      if (className) el.className = className;
      el.textContent = text;
      parent.appendChild(el);
      return el;
    }
    function renderLogCards(rows) {
      const showHttp = $('showHttpLogs')?.checked;
      const filtered = rows.filter(r => showHttp || r.event !== 'http').slice(-80).reverse();
      const view = $('logView');
      view.innerHTML = '';
      $('logHint').textContent = `${currentLog} · ${filtered.length} cards${showHttp ? '' : ' · HTTP 轮询已隐藏'}`;
      if (!filtered.length) {
        appendText(view, 'div', 'status', '暂无可展示日志。');
        return;
      }
      filtered.forEach(rec => {
        const meta = logSummary(rec);
        const card = document.createElement('article');
        card.className = `logCard ${logCardClass(rec)}`;
        const top = document.createElement('div');
        top.className = 'logTop';
        appendText(top, 'div', 'logTitle', meta.title);
        appendText(top, 'div', 'logTime', rec.ts || '');
        card.appendChild(top);
        appendText(card, 'div', 'logSummary', meta.summary);
        const chips = document.createElement('div');
        chips.className = 'chips';
        (meta.chips || []).filter(Boolean).slice(0, 6).forEach(c => appendText(chips, 'span', 'chip', c));
        card.appendChild(chips);
        const details = document.createElement('details');
        const summary = document.createElement('summary');
        summary.textContent = '原始详情';
        const pre = document.createElement('pre');
        pre.textContent = JSON.stringify(rec, null, 2);
        details.appendChild(summary);
        details.appendChild(pre);
        card.appendChild(details);
        view.appendChild(card);
      });
    }
    function fmt(v, digits=4) {
      return Number.isFinite(v) ? v.toFixed(digits) : '--';
    }
    function colorFor(row) {
      if (row.score_anomaly) return '#7c3aed';
      if (row.decision === 'ACCEPTED') return '#087443';
      if (row.decision === 'REJECTED') return '#b54708';
      return '#b42318';
    }
    function drawScoreChart(rows) {
      const svg = $('scoreChart');
      const w = 900, h = 280, m = {l:58, r:22, t:22, b:42};
      svg.innerHTML = '';
      if (!rows.length) {
        svg.innerHTML = '<text x="450" y="140" text-anchor="middle" fill="#667085" font-size="14">暂无迭代数据</text>';
        $('latestScore').textContent = '--';
        $('bestScore').textContent = '--';
        $('iterCount').textContent = '0';
        $('chartSubtitle').textContent = 'runner.py once 后会自动出现曲线';
        return;
      }
      const valid = rows.filter(r => Number.isFinite(Number(r.score)));
      const latest = rows[rows.length - 1];
      const best = valid.reduce((acc, r) => Math.max(acc, Number(r.score)), -Infinity);
      $('latestScore').textContent = fmt(Number(latest.score));
      $('bestScore').textContent = fmt(best);
      $('iterCount').textContent = String(rows.length);
      $('chartSubtitle').textContent = `latest: #${latest.iter_id} ${latest.decision || latest.status || ''} · ${latest.score_version || '--'} · horizon=${latest.horizon ?? '--'} · sharpe=${fmt(Number(latest.sharpe), 3)}`;
      const xs = rows.map(r => Number(r.iter_id)).filter(Number.isFinite);
      const ys = valid.flatMap(r => [Number(r.score), Number(r.best_score)]).filter(Number.isFinite);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      let minY = Math.min(...ys), maxY = Math.max(...ys);
      if (minY === maxY) { minY -= 1; maxY += 1; }
      const padY = (maxY - minY) * 0.12 || 1;
      minY -= padY; maxY += padY;
      const x = v => m.l + (maxX === minX ? 0.5 : (v - minX) / (maxX - minX)) * (w - m.l - m.r);
      const y = v => h - m.b - (v - minY) / (maxY - minY) * (h - m.t - m.b);
      const el = (name, attrs, text) => {
        const node = document.createElementNS('http://www.w3.org/2000/svg', name);
        for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
        if (text !== undefined) node.textContent = text;
        svg.appendChild(node);
        return node;
      };
      el('rect', {x:0, y:0, width:w, height:h, fill:'#fbfcfe'});
      for (let i = 0; i <= 4; i++) {
        const value = minY + (maxY - minY) * i / 4;
        const yy = y(value);
        el('line', {x1:m.l, y1:yy, x2:w-m.r, y2:yy, stroke:'#e5e7eb', 'stroke-width':1});
        el('text', {x:m.l-10, y:yy+4, 'text-anchor':'end', fill:'#667085', 'font-size':11}, fmt(value, 2));
      }
      el('line', {x1:m.l, y1:h-m.b, x2:w-m.r, y2:h-m.b, stroke:'#98a2b3', 'stroke-width':1});
      el('line', {x1:m.l, y1:m.t, x2:m.l, y2:h-m.b, stroke:'#98a2b3', 'stroke-width':1});
      const scorePts = valid.map(r => `${x(Number(r.iter_id))},${y(Number(r.score))}`).join(' ');
      if (scorePts) el('polyline', {points:scorePts, fill:'none', stroke:'#111827', 'stroke-width':2.2, 'stroke-linejoin':'round', 'stroke-linecap':'round'});
      const bestPts = valid.map(r => `${x(Number(r.iter_id))},${y(Number(r.best_score))}`).join(' ');
      if (bestPts) el('polyline', {points:bestPts, fill:'none', stroke:'#1d4ed8', 'stroke-width':2, 'stroke-dasharray':'6 4'});
      rows.forEach(row => {
        const score = Number(row.score);
        if (!Number.isFinite(score)) return;
        const cx = x(Number(row.iter_id)), cy = y(score);
        const c = el('circle', {cx, cy, r: row.score_anomaly ? 6 : 5, fill:colorFor(row), stroke:'#fff', 'stroke-width':1.5});
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = `#${row.iter_id} ${row.decision || row.status}\nscore ${fmt(score)}\nbest ${fmt(Number(row.best_score))}\nrank_ic_ir ${fmt(Number(row.rank_ic_ir), 3)}\nsharpe ${fmt(Number(row.sharpe), 3)}`;
        c.appendChild(title);
        el('text', {x:cx, y:h-18, 'text-anchor':'middle', fill:'#667085', 'font-size':10}, row.iter_id);
      });
    }
    async function refreshScores() {
      const rows = await api('/api/scores');
      drawScoreChart(rows);
    }
    async function refreshRuns() {
      const rows = await api('/api/runs?limit=80');
      renderRunCards(rows);
    }
    async function appendManualLog() {
      await api('/api/logs', {method:'POST', body:JSON.stringify({kind:$('manual_kind').value, text:$('manual_text').value})});
      $('manual_text').value = '';
      await refreshLog();
    }
    async function refreshAll(){ await Promise.all([refreshStatus(), refreshConfig(), refreshLog(), refreshScores(), refreshRuns(), refreshMemory()]); }
    refreshAll();
    setInterval(() => { refreshStatus(); refreshLog(); refreshScores(); refreshRuns(); refreshMemory(); }, 3000);
  </script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def _json(self, data: Any, status: int = 200) -> None:
        raw = json.dumps(json_safe(data), ensure_ascii=False, default=str, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        try:
            if self.path == "/" or self.path.startswith("/index.html"):
                raw = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif self.path.startswith("/api/status"):
                with STATE_LOCK:
                    runtime = dict(RUNTIME)
                self._json({"runtime": runtime})
            elif self.path.startswith("/api/config"):
                self._json(load_config(include_secret=False))
            elif self.path.startswith("/api/memory"):
                self._json(load_memory())
            elif self.path.startswith("/api/bottleneck"):
                self._json(detect_bottlenecks())
            elif self.path.startswith("/api/scores"):
                self._json(score_series())
            elif self.path.startswith("/api/runs"):
                query = self.path.split("?", 1)[1] if "?" in self.path else ""
                params = dict(part.split("=", 1) for part in query.split("&") if "=" in part)
                self._json(journal_runs(limit=int(params.get("limit", "80"))))
            elif self.path.startswith("/api/logs"):
                query = self.path.split("?", 1)[1] if "?" in self.path else ""
                params = dict(part.split("=", 1) for part in query.split("&") if "=" in part)
                kind = params.get("kind", "audit")
                limit = int(params.get("limit", "200"))
                if kind not in LOG_FILES:
                    self._json({"error": "unknown log kind"}, 400)
                else:
                    self._json(read_log(kind, limit=limit))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:
            self._json({"error": str(exc)}, 500)

    def do_POST(self) -> None:
        try:
            data = self._read_json()
            if self.path.startswith("/api/config"):
                save_config(data)
                self._json(load_config(include_secret=False), 200)
            elif self.path.startswith("/api/memory/reset"):
                self._json(reset_memory(), 200)
            elif self.path.startswith("/api/test-api"):
                cfg = {**load_config(), **data}
                result = test_api_config(cfg)
                append_log("audit", "api_connection_test", {
                    "ok": result["ok"],
                    "base_url": result["base_url"],
                    "model": result["model"],
                    "models": result["models"],
                    "chat_model": result.get("chat_model"),
                    "usage": result.get("usage"),
                })
                self._json(result, 200)
            elif self.path.startswith("/api/start"):
                start_loop()
                self._json({"ok": True})
            elif self.path.startswith("/api/stop"):
                stop_loop()
                self._json({"ok": True})
            elif self.path.startswith("/api/logs"):
                kind = data.get("kind", "audit")
                if kind not in LOG_FILES:
                    self._json({"error": "unknown log kind"}, 400)
                else:
                    append_log(kind, "manual_note", {"text": data.get("text", "")})
                    self._json({"ok": True})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:
            self._json({"error": str(exc)}, 500)

    def log_message(self, fmt: str, *args: Any) -> None:
        append_log("audit", "http", {"client": self.client_address[0], "message": fmt % args})


def main() -> None:
    ensure_state()
    host = os.environ.get("AUTOALPHA_SERVICE_HOST", "127.0.0.1")
    port = int(os.environ.get("AUTOALPHA_SERVICE_PORT", "8765"))
    append_log("audit", "service_started", {"host": host, "port": port})
    print(f"AutoAlpha service: http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
