"""Why Owl: a kid-safe "why?" explainer powered by Gemma 4.

Run:   export GEMINI_API_KEY=...   then   python server.py   and open http://localhost:8000

Backends (LLM_BACKEND env var):
  gemini  (default)  hosted Gemma 4 via the Gemini API
  ollama             local Gemma via Ollama (set OLLAMA_MODEL, e.g. gemma4:e4b)
  fake               canned replies, for tests only
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"
BACKEND = os.environ.get("LLM_BACKEND", "gemini").lower()
GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "gemma-4-26b-a4b-it")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
PORT = int(os.environ.get("PORT", "8000"))
MAX_BODY = 8 * 1024 * 1024  # webcam snapshots are well under this

AGE_GUIDE = {
    "3-5": "a 3-5 year old. Very short sentences, everyday words, compare to things at home (bath, toys, snacks). 2-3 sentences.",
    "6-8": "a 6-8 year old. Simple words and one concrete comparison or example. 3-4 sentences.",
    "9-12": "a 9-12 year old. Explain the real mechanism simply; you may use one science word if you explain it. 4-5 sentences.",
}

ASK_SYSTEM = """You are Why Owl, a warm, playful owl who answers children's "why" questions.

You are talking with {age_guide}

Rules:
- Be accurate. Only state well-established facts. If scientists don't know yet, say so and make that sound exciting.
- Never make the child feel a question was silly.
- Never ask for personal details (name, address, school, phone). If the child shares them, don't repeat them.
- No scary or gory details, no brands, no links, no emojis.
- Sensitive topics (bodies and where babies come from, death, illness, violence, scary news, family problems): give a short, gentle, true answer without detail, suggest talking with a grown-up they trust (parent, teacher, family member), and set "sensitive" to a short neutral topic label such as "death" or "where babies come from".
- If the child says someone is hurting them, or they feel very sad or unsafe, kindly tell them to tell a trusted grown-up right away, and set "sensitive" to "safety concern".
- If an image is included, base your answer on what is in it.
- If the child is answering your last follow-up question, react warmly to their thinking (gently correct it if it's wrong), explain a little more, then ask a new follow-up.
- Always end with ONE short follow-up question that gets the child to guess, notice, or try something. Not a yes/no question.

Reply with ONLY a JSON object, no code fences:
{{"answer": "...", "follow_up": "...", "sensitive": null}}"""

DIGEST_SYSTEM = """You help a busy parent keep up with their child's curiosity.
You get the questions their child ({age} years old) asked Why Owl today, with the owl's answers.

Reply with ONLY a JSON object, no code fences:
{{"summary": "one sentence on what the child was curious about today",
 "starters": ["three dinner-table conversation starters that build on these questions"],
 "activity": "one simple, safe, 10-minute at-home activity or experiment related to the top topic, using common household items",
 "sensitive_note": "if any question is marked sensitive: name the topic(s) neutrally and give one tip for talking about it at the child's level; otherwise null"}}"""


# ---------------------------------------------------------------- model calls

_gemini_client = None


def _gemini(system: str, turns: list[dict], image: dict | None, temperature: float) -> str:
    global _gemini_client
    from google import genai
    from google.genai import types

    if _gemini_client is None:
        _gemini_client = genai.Client()
    contents = []
    for i, t in enumerate(turns):
        parts = []
        if image and i == len(turns) - 1:
            parts.append(types.Part.from_bytes(data=base64.b64decode(image["data"]), mime_type=image.get("mime", "image/jpeg")))
        parts.append(types.Part.from_text(text=t["text"]))
        contents.append(types.Content(role=t["role"], parts=parts))

    def call(thinking: bool):
        cfg = {"system_instruction": system, "temperature": temperature}
        if thinking:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level="minimal")  # fast answers
        return _gemini_client.models.generate_content(
            model=GEMMA_MODEL, contents=contents, config=types.GenerateContentConfig(**cfg)
        ).text

    try:
        return call(thinking=True)
    except Exception as e:  # older SDKs or models without thinking control
        if "think" in str(e).lower():
            return call(thinking=False)
        raise


def _ollama(system: str, turns: list[dict], image: dict | None, temperature: float) -> str:
    messages = [{"role": "system", "content": system}]
    for i, t in enumerate(turns):
        msg = {"role": "assistant" if t["role"] == "model" else "user", "content": t["text"]}
        if image and i == len(turns) - 1:
            msg["images"] = [image["data"]]
        messages.append(msg)
    payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False, "format": "json",
               "options": {"temperature": temperature}}
    host = OLLAMA_HOST if OLLAMA_HOST.startswith("http") else f"http://{OLLAMA_HOST}"
    req = urllib.request.Request(f"{host.rstrip('/')}/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.load(resp)
    if "error" in data:
        raise RuntimeError(data["error"])
    return data["message"]["content"]


def _fake(system: str, turns: list[dict], image: dict | None, temperature: float) -> str:
    if system.startswith("You help a busy parent"):
        return json.dumps({"summary": "Today was all about the sky.",
                           "starters": ["What color would the sky be on Mars?", "Why are sunsets orange?", "What else bends light?"],
                           "activity": "Shine a flashlight through a glass of water with a few drops of milk.",
                           "sensitive_note": None})
    q = turns[-1]["text"].lower()
    sensitive = "death" if "die" in q else None
    return "```json\n" + json.dumps({"answer": "Sunlight is made of all colors, and the air scatters blue light the most, so blue reaches your eyes from every direction.",
                                     "follow_up": "What color do you think the sky is at sunset, and why?",
                                     "sensitive": sensitive}) + "\n```"


BACKENDS = {"gemini": _gemini, "ollama": _ollama, "fake": _fake}


def parse_json(text: str, required: tuple[str, ...]) -> dict:
    """Models sometimes wrap JSON in fences or add text; extract the object, or fall back to raw text."""
    text = (text or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict) and all(k in data for k in required):
                return data
        except json.JSONDecodeError:
            pass
    return {"answer": re.sub(r"```\w*", "", text).strip(), "follow_up": "", "sensitive": None, "_unparsed": True}


def ask(body: dict) -> dict:
    age = body.get("age") if body.get("age") in AGE_GUIDE else "6-8"
    question = (body.get("question") or "").strip()[:500]
    image = body.get("image") if isinstance(body.get("image"), dict) and body["image"].get("data") else None
    if not question:
        question = "What is this, and why is it like that?" if image else ""
    if not question:
        raise ValueError("Empty question")
    history = [t for t in body.get("history", [])[-8:]
               if isinstance(t, dict) and t.get("role") in ("user", "model") and t.get("text")]
    turns = history + [{"role": "user", "text": question}]
    raw = BACKENDS[BACKEND](ASK_SYSTEM.format(age_guide=AGE_GUIDE[age]), turns, image, 0.6)
    data = parse_json(raw, ("answer", "follow_up"))
    return {"answer": str(data.get("answer", "")).strip(),
            "follow_up": str(data.get("follow_up") or "").strip(),
            "sensitive": data.get("sensitive") or None}


def digest(body: dict) -> dict:
    items = [i for i in body.get("items", []) if isinstance(i, dict) and i.get("q")][-30:]
    if not items:
        raise ValueError("No questions yet")
    age = body.get("age") if body.get("age") in AGE_GUIDE else "6-8"
    listing = "\n".join(
        f"- Q: {i['q']}\n  A: {i.get('a', '')}" + (f"\n  [sensitive: {i['sensitive']}]" if i.get("sensitive") else "")
        for i in items
    )
    raw = BACKENDS[BACKEND](DIGEST_SYSTEM.format(age=age), [{"role": "user", "text": listing}], None, 0.7)
    data = parse_json(raw, ("summary", "starters"))
    if data.get("_unparsed"):
        return {"summary": data["answer"], "starters": [], "activity": "", "sensitive_note": None}
    return data


# ---------------------------------------------------------------- HTTP

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_request(self, code="-", size="-"):  # keep the terminal quiet except for errors
        try:
            failed = int(code) >= 400
        except (TypeError, ValueError):
            failed = False
        if failed:
            super().log_request(code, size)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        routes = {"/api/ask": ask, "/api/digest": digest}
        if self.path not in routes:
            return self._send_json(404, {"error": "Not found"})
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY:
            return self._send_json(413, {"error": "Image too large"})
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            return self._send_json(200, routes[self.path](body))
        except ValueError as e:
            return self._send_json(400, {"error": str(e)})
        except Exception as e:
            print(f"[error] {self.path}: {e}", file=sys.stderr)
            return self._send_json(500, {"error": str(e)})


def main() -> None:
    if BACKEND not in BACKENDS:
        sys.exit(f"Unknown LLM_BACKEND={BACKEND!r}. Use gemini, ollama or fake.")
    if BACKEND == "gemini" and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit("Set your key first:  export GEMINI_API_KEY='...'  (https://aistudio.google.com/apikey)")
    model = {"gemini": GEMMA_MODEL, "ollama": OLLAMA_MODEL, "fake": "canned replies"}[BACKEND]
    print(f"🦉 Why Owl on http://localhost:{PORT}  ·  backend={BACKEND}  ·  model={model}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
