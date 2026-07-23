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
REM NOTE: uses "python -m pip" (not bare "pip") because on plain Windows
REM Python installs, pip.exe often isn't on PATH even though python.exe is.
REM --break-system-packages was dropped: it's a Linux/PEP-668 flag that some
REM Windows pip versions don't recognize, which was silently aborting the
REM whole install before any packages landed (that's what broke this before).
echo 📦 Installing dependencies...
python -m pip install -r backend\requirements.txt -q
if errorlevel 1 (
    echo.
    echo ❌ Dependency install failed — see errors above. Fix them and re-run this script.
    pause
    exit /b 1
)

REM Install Whisper (audio/video transcription support — optional, app runs fine without it)
echo 🎙️ Installing openai-whisper for audio/video support (this may take a few minutes)...
python -m pip install openai-whisper -q

REM Start backend — kept in a window that stays open (cmd /k) so any crash
REM/error is visible instead of the window vanishing before you can read it.
echo 🚀 Starting FastAPI backend...
cd backend
start "AI File Intelligence Bot - Backend" cmd /k python main.py
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
