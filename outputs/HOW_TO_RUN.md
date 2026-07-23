# How to Run — AI File Intelligence Bot (CR-01, offline multi-user)

## 1. One-time prerequisites

- **Ollama installed and running** on this machine, with at least one model pulled:
  ```
  ollama --version
  ollama list
  ollama pull llama3.1:8b        (skip if you already have a model)
  ollama serve                    (skip if Ollama already runs as a service)
  ```
- Python installed (confirmed working: `Python314` on this machine).

## 2. Start the app

Easiest: double-click **`start_windows.bat`** in the project root. It now:
- Installs dependencies with `python -m pip` (fixed — the old `--break-system-packages` flag was silently breaking the install on this machine)
- Fails loudly with a visible error if the install breaks, instead of continuing with missing packages
- Opens the backend in a window that **stays open** on crash so you can read the error
- Opens `http://localhost:8000` in your browser automatically

Manual alternative, if you ever want to see everything in one terminal:
```
cd "C:\Users\user\Documents\Claude\Projects\AI FILE INTELLIGENCE BOT\backend"
python -m pip install -r requirements.txt
python main.py
```
Then open `http://localhost:8000`.

## 3. Log in

- Username: `admin`
- Password: `ChangeMe123!`

Change this password from the Admin panel after logging in — it's a bootstrap default, not meant to stay in place.

## 4. If chat answers look wrong or slow

- First message after starting the server can take 30–90s (or more) while the model loads into memory — this is normal, not a bug. Wait for it rather than resubmitting (sending a second question while the first is still processing makes both slower, since Ollama handles one request at a time per model).
- Check the **Models** tab to confirm Ollama shows as reachable and a model is marked active.
- If it still falls back to a low-quality answer, check the backend terminal window for `WARNING` lines — they name exactly which provider failed and why.

## 5. Optional: cloud fallback providers

Only needed if you want a backup when Ollama is down. Edit `backend\.env`:
- `MISTRAL_API_KEY` — get one at console.mistral.ai
- `GOOGLE_API_KEY` — get one at aistudio.google.com/apikey (for Gemini)

Leave blank to skip — Ollama alone is enough to run fully offline.

## Reference

Full architecture write-up: `CR-01_OFFLINE_PLATFORM.md` in the project root.
Architecture diagram: `architecture.jpg` in this `outputs` folder.
