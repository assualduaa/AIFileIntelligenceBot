@echo off
REM ── AI File Intelligence Bot — Windows startup ──────────────────────
echo 🧠 AI File Intelligence Bot — Starting...

cd /d "%~dp0"

REM Copy .env if missing
if not exist .env (
    copy .env.example .env
    echo 📝 Created .env — add your OPENAI_API_KEY if you have one.
)

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r backend\requirements.txt --break-system-packages -q

REM Install Whisper (audio/video transcription support)
echo 🎙️ Installing openai-whisper for audio/video support (this may take a few minutes)...
pip install openai-whisper --break-system-packages -q

REM Start backend
echo 🚀 Starting FastAPI backend...
cd backend
start "" python main.py
cd ..

timeout /t 5 /nobreak > nul

echo.
echo ══════════════════════════════════════════
echo   🧠 AI File Intelligence Bot is LIVE
echo   Backend API:  http://localhost:8000
echo   Frontend UI:  http://localhost:8000
echo   API Docs:     http://localhost:8000/docs
echo ══════════════════════════════════════════
echo.
echo Opening browser...
start http://localhost:8000

pause
