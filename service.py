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


def call_openai_compatible(cfg: dict[str, Any], messages: list[dict[str, str]]) -> dict[str, Any]:
    if not cfg.get("base_url") or not cfg.get("api_key") or not cfg.get("model"):
        raise RuntimeError("API config is incomplete: base_url, api_key, and model are required.")
    body = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": cfg.get("temperature", 0.2),
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        normalize_base_url(cfg["base_url"]),
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API HTTP {exc.code}: {detail}") from exc
    content = data["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model did not return JSON. Content starts with: {content[:500]}") from exc


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


def one_iteration() -> None:
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
    ALPHA_PATH.write_text(alpha_py, encoding="utf-8")
    append_log("action", "alpha_replaced", {"iteration": iteration, "backup": str(backup_path)})

    py = str(PYTHON if PYTHON.exists() else "python")
    with STATE_LOCK:
        RUNTIME["status"] = "evaluating"
    code, output = run_cmd([py, str(RUNNER), "once"], timeout=3600)
    rec = last_run_record()
    append_log("delivery", "iteration_evaluated", {
        "iteration": iteration,
        "returncode": code,
        "runner_tail": output[-4000:],
        "run_record": rec,
    })
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
            one_iteration()
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
    main { padding:20px 24px; display:grid; grid-template-columns:360px 1fr; gap:20px; }
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
    @media (max-width: 900px) { main { grid-template-columns:1fr; } .log { height:480px; } }
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
        <button onclick="refreshAll()">刷新</button>
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
    <section>
      <div class="tabs">
        <button id="tab_audit" class="active" onclick="setLog('audit')">审计日志</button>
        <button id="tab_action" onclick="setLog('action')">行动日志</button>
        <button id="tab_research" onclick="setLog('research')">研究日志</button>
        <button id="tab_delivery" onclick="setLog('delivery')">交付日志</button>
      </div>
      <div class="log"><pre id="logView"></pre></div>
    </section>
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
    async function appendManualLog() {
      await api('/api/logs', {method:'POST', body:JSON.stringify({kind:$('manual_kind').value, text:$('manual_text').value})});
      $('manual_text').value = '';
      await refreshLog();
    }
    async function refreshAll(){ await Promise.all([refreshStatus(), refreshConfig(), refreshLog()]); }
    refreshAll();
    setInterval(() => { refreshStatus(); refreshLog(); }, 3000);
  </script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def _json(self, data: Any, status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
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
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif self.path.startswith("/api/status"):
                with STATE_LOCK:
                    runtime = dict(RUNTIME)
                self._json({"runtime": runtime})
            elif self.path.startswith("/api/config"):
                self._json(load_config(include_secret=False))
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
