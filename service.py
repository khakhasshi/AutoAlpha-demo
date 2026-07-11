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
    "enabled": False,
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
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    append_log("audit", "config_saved", {
        "base_url": cfg.get("base_url"),
        "model": cfg.get("model"),
        "has_api_key": bool(cfg.get("api_key")),
        "iteration_sleep_sec": cfg.get("iteration_sleep_sec"),
        "auto_commit_accepted": cfg.get("auto_commit_accepted"),
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
    rows = []
    best = float("-inf")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        score = rec.get("score")
        if isinstance(score, int | float):
            best = max(best, float(score))
            best_score = best
        else:
            best_score = None if best == float("-inf") else best
        rows.append({
            "iter_id": rec.get("iter_id"),
            "ts": rec.get("ts"),
            "score": score,
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


def build_prompt() -> list[dict[str, str]]:
    program = read_text("program.md")
    evaluation = read_text("evaluation.md")
    alpha = read_text("alpha.py", max_chars=40000)
    status = latest_runner_status()
    system = (
        "You are an autonomous quantitative research coding agent inside the AutoAlpha demo. "
        "Your only allowed code change is replacing alpha.py. Obey program.md and evaluation.md. "
        "Make exactly one research change per iteration. Return strict JSON only."
    )
    user = f"""
We need the next AutoAlpha iteration.

Rules:
- Return a JSON object with keys: summary, research_note, alpha_py.
- alpha_py must be the complete replacement content for alpha.py.
- Do not modify prepare.py, runner.py, metrics.py, evaluation.md, program.md, data files, or service files.
- Do not include any source strings forbidden by program.md inside alpha_py.
- Keep the public contract: HORIZON, LABEL_KIND, ITER_NOTE, FACTORS, run(train_panel, val_panel).
- Use only train data to estimate any fitted parameters. Validation is for runner evaluation only.
- Prefer a single clear hypothesis likely to improve prepare.primary_score, unless the latest logs suggest a score anomaly.

Current runner status:
{status}

program.md:
{program}

evaluation.md:
{evaluation}

current alpha.py:
{alpha}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


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

    messages = build_prompt()
    append_log("research", "context_built", {"iteration": iteration, "message_count": len(messages)})
    proposal = call_openai_compatible(cfg, messages)
    alpha_py = proposal.get("alpha_py")
    if not isinstance(alpha_py, str) or "def run(" not in alpha_py or "ITER_NOTE" not in alpha_py:
        raise RuntimeError("Proposal missing a complete alpha_py with ITER_NOTE and run().")
    append_log("research", "proposal_received", {
        "iteration": iteration,
        "summary": proposal.get("summary"),
        "research_note": proposal.get("research_note"),
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
    if rec and rec.get("score_anomaly"):
        append_log("audit", "paused_for_score_anomaly", {
            "iteration": iteration,
            "score": rec.get("score"),
            "decision": rec.get("decision"),
            "score_anomaly": rec.get("score_anomaly"),
        })
        with STATE_LOCK:
            RUNTIME["status"] = "waiting_for_human_score_anomaly"
            RUNTIME["last_finished_at"] = now_iso()
        return False
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
        if final_status not in {"waiting_for_human_score_anomaly"}:
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
    .log { height:620px; overflow:auto; background:#0b1020; color:#dbeafe; border-radius:8px; padding:14px; }
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
      <div class="row" style="margin-top:14px">
        <button class="primary" onclick="saveConfig()">保存配置</button>
        <button onclick="fillLmStudio()">填入 LM Studio</button>
        <button onclick="testConnection()">测试连接</button>
        <button onclick="refreshAll()">刷新</button>
      </div>
      <pre id="testResult" class="muted" style="margin-top:12px"></pre>
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
        <div class="tabs">
          <button id="tab_audit" class="active" onclick="setLog('audit')">审计日志</button>
          <button id="tab_action" onclick="setLog('action')">行动日志</button>
          <button id="tab_research" onclick="setLog('research')">研究日志</button>
          <button id="tab_delivery" onclick="setLog('delivery')">交付日志</button>
        </div>
        <div class="log"><pre id="logView"></pre></div>
      </section>
    </div>
  </main>
  <script>
    let currentLog = 'audit';
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
        sleeping: '本轮已交付，等待下一轮迭代间隔。',
        stopping: '正在停止后台循环。',
        error_sleeping: '上一轮出错，服务短暂停顿后可复查日志。',
        waiting_for_human_score_anomaly: '发现 score anomaly，已暂停等待人工复核。'
      };
      $('flowStatus').textContent = labels[status] || `当前状态：${status}`;
      $('flowPill').textContent = status;
    }
    async function refreshConfig() {
      const cfg = await api('/api/config');
      for (const k of ['base_url','api_key','model','temperature','iteration_sleep_sec']) $(k).value = cfg[k] || '';
      $('auto_commit_accepted').checked = !!cfg.auto_commit_accepted;
    }
    async function saveConfig() {
      await api('/api/config', {method:'POST', body:JSON.stringify({
        base_url:$('base_url').value, api_key:$('api_key').value, model:$('model').value,
        temperature:parseFloat($('temperature').value || '0.2'),
        iteration_sleep_sec:parseInt($('iteration_sleep_sec').value || '5', 10),
        auto_commit_accepted:$('auto_commit_accepted').checked
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
      $('logView').textContent = rows.map(r => JSON.stringify(r, null, 2)).join('\n\n');
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
      $('chartSubtitle').textContent = `latest: #${latest.iter_id} ${latest.decision || latest.status || ''} · horizon=${latest.horizon ?? '--'} · sharpe=${fmt(Number(latest.sharpe), 3)}`;
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
    async function appendManualLog() {
      await api('/api/logs', {method:'POST', body:JSON.stringify({kind:$('manual_kind').value, text:$('manual_text').value})});
      $('manual_text').value = '';
      await refreshLog();
    }
    async function refreshAll(){ await Promise.all([refreshStatus(), refreshConfig(), refreshLog(), refreshScores()]); }
    refreshAll();
    setInterval(() => { refreshStatus(); refreshLog(); refreshScores(); }, 3000);
  </script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def _json(self, data: Any, status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
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
            elif self.path.startswith("/api/scores"):
                self._json(score_series())
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
