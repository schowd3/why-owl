# 🦉 Why Owl — endless "why?" questions, handled

Kids ask "why?" all day. Why Owl answers out loud at their age level, then asks them a follow-up question to keep them thinking. The parent gets a summary of what their kid was curious about, with dinner-table conversation starters and an at-home activity.

Powered by **Gemma 4** (Google's open-weight model) through the Gemini API, or locally through Ollama.
Built at Hacktoberfest Hack Day Astoria (Oct 21, 2026).

## Features

- **Ask out loud.** Tap the mic. Uses the browser's speech recognition (Chrome or Safari).
- **Hear the answer.** The browser reads it aloud, and you can replay it.
- **📷 Show me.** Snap a webcam picture of a leaf, bug or toy and ask about it. Gemma 4 answers from the image.
- **Age levels: 3–5, 6–8 and 9–12.** These change vocabulary, length and depth.
- **A follow-up after every answer.** The kid answers it, and the owl reacts to their reasoning.
- **👪 Parent corner.** It lists today's questions and flags sensitive topics. "Make my parent summary" generates conversation starters and a 10-minute activity.

## Safety and privacy choices

- The system prompt allows only well-established facts and has the owl say "scientists don't know yet" when that's true.
- On sensitive topics (death, bodies, scary news, family problems), the owl gives a short, gentle, true answer, suggests a trusted grown-up, and flags the topic for the parent.
- If a child says someone is hurting them or they feel unsafe, the owl tells them to tell a trusted grown-up right away.
- The owl never asks for personal details.
- The child's name is shown on screen only and is **never sent to the model**.
- Nothing is stored on the server. The question list lives in the page's memory and is gone when the tab closes.

## Run it (macOS)

```bash
cd why-owl
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="..."        # https://aistudio.google.com/apikey
python server.py                   # open http://localhost:8000 in Chrome
```

Allow the microphone and camera when the browser asks.

### Run fully local with Ollama

This needs roughly 16 GB of RAM or more.

```bash
ollama pull gemma4:e4b
LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e4b python server.py
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BACKEND` | `gemini` | `gemini`, `ollama`, or `fake` (canned replies, for UI testing only) |
| `GEMMA_MODEL` | `gemma-4-26b-a4b-it` | Gemini API model, e.g. `gemma-4-31b-it` |
| `OLLAMA_MODEL` | `gemma4:e4b` | Local model name |
| `PORT` | `8000` | Server port |

## How it works

```
browser (speech in/out, webcam)  →  POST /api/ask {question, age, last 8 turns, image?}
                                     → Gemma 4 returns {"answer", "follow_up", "sensitive"}
parent corner                     →  POST /api/digest {today's Q&A}
                                     → Gemma 4 returns {summary, starters[], activity, sensitive_note}
```

`server.py` uses only the Python standard library plus `google-genai`. The page is a single HTML file with no build step.

## Limitations and next steps

- This is a hackathon prototype, not a reviewed children's product. A real launch needs child-safety review, COPPA compliance and age-appropriate content testing.
- Speech recognition quality depends on the browser and room noise.
- Next steps:
  - Use ShieldGemma to screen inputs and outputs.
  - Add more languages so kids can practice in a heritage language.
  - Add an optional weekly email digest for parents.
