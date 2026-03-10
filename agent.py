"""
FangYuan SN36 Web Agent
Bittensor Subnet 36 (Autoppia) — web task automation via LLM-driven actions.
"""
import os
import re
import json
import httpx
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-4o-mini")  # cost-efficient default


# ── Models ────────────────────────────────────────────────────────────────────

class ActRequest(BaseModel):
    task_id:      str
    prompt:       str | None = None
    task_prompt:  str | None = None
    url:          str | None = None
    snapshot_html:str | None = None
    screenshot:   str | None = None   # base64 PNG, not used in LLM call (cost)
    step_index:   int = 0
    history:      list = []
    model:        str | None = None
    target_hint:  str | None = None


class ActResponse(BaseModel):
    actions: list[dict]
    metrics: dict = {}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/act", response_model=ActResponse)
@app.post("/step", response_model=ActResponse)
async def act(req: ActRequest):
    task = req.prompt or req.task_prompt or ""
    model = req.model or LLM_MODEL

    # Build compact HTML summary (first 3000 chars to stay under cost limit)
    html_summary = (req.snapshot_html or "")[:3000]
    history_summary = _fmt_history(req.history[-5:])  # last 5 steps only

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": _build_prompt(task, req.url, html_summary, history_summary, req.step_index, req.target_hint)},
    ]

    try:
        actions = await _call_llm(model, messages, req.task_id)
    except Exception as e:
        # Fallback: do nothing and let the validator move on
        print(f"[agent] LLM error: {e}")
        actions = [{"type": "WaitAction", "duration_ms": 500}]

    return ActResponse(actions=actions)


# ── LLM ───────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise web automation agent. Given a task, current page URL, HTML snapshot, and action history, decide the next browser action to complete the task.

Output ONLY valid JSON — an array of actions. No explanation, no markdown.

Action types (ONLY return 1 action per step — only the first is executed):
- ClickAction:    {"type":"ClickAction","selector":{"css":"#id or .class"}}
- TypeAction:     {"type":"TypeAction","selector":{"css":"input[name=q]"},"text":"value to type"}
- NavigateAction: {"type":"NavigateAction","url":"https://..."}
- SelectDropDownOptionAction: {"type":"SelectDropDownOptionAction","selector":{"css":"select#id"},"text":"Option Label"}
- ScrollAction:   {"type":"ScrollAction","direction":"down","amount":300}
- WaitAction:     {"type":"WaitAction","duration_ms":500}
- done:           {"type":"done"}

Rules:
- Return exactly 1 action in the array
- Prefer specific CSS selectors (id > name attr > class)
- If task is complete, return [{"type":"done"}]
- If stuck (same action 3 times), try NavigateAction or ScrollAction to get unstuck
- Never exceed 12 steps (current step_index is your guide)
- Stay under $0.05 total LLM cost per task
"""


def _build_prompt(task, url, html, history, step, target_hint):
    parts = [
        f"Task: {task}",
        f"Step: {step}",
        f"URL: {url or 'unknown'}",
    ]
    if target_hint:
        parts.append(f"Hint: {target_hint}")
    if history:
        parts.append(f"History:\n{history}")
    if html:
        parts.append(f"HTML (truncated):\n{html}")
    parts.append("Next action(s) as JSON array:")
    return "\n\n".join(parts)


def _fmt_history(history):
    if not history:
        return ""
    lines = []
    for h in history:
        t = h.get("type", "?")
        sel = h.get("selector", {}).get("value", "") if isinstance(h.get("selector"), dict) else ""
        txt = h.get("text", "") or h.get("url", "")
        lines.append(f"  {t} {sel} {txt}".strip())
    return "\n".join(lines)


async def _call_llm(model: str, messages: list, task_id: str) -> list:
    headers = {
        "Authorization":  f"Bearer {OPENAI_API_KEY}",
        "Content-Type":   "application/json",
        "IWA-Task-ID":    task_id,  # required by validator gateway
    }
    payload = {
        "model":    model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_tokens": 300,
        "temperature": 0.0,
    }
    async with httpx.AsyncClient(base_url=OPENAI_BASE_URL, timeout=30) as c:
        r = await c.post("/v1/chat/completions", json=payload, headers=headers)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]

    return _parse_actions(content)


def _parse_actions(text: str) -> list:
    text = text.strip()
    # Try direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        # Unwrap {"actions": [...]} or any top-level list value
        for v in parsed.values():
            if isinstance(v, list):
                return v
        return []
    except json.JSONDecodeError:
        pass
    # Fallback: extract first JSON array
    m = re.search(r'\[.*?\]', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return []
