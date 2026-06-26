#!/bin/bash
# ── AI File Intelligence Bot — startup script ─────────────────────────
set -e

echo "🧠 AI File Intelligence Bot — Starting..."
cd "$(dirname "$0")"

# 1. Create .env if missing
if [ ! -f .env ]; then
  cp .env.example .env
  echo "📝 Created .env from template. Add your OPENAI_API_KEY if you have one."
fi

# 2. Install Python deps
echo "📦 Installing dependencies..."
pip install -r backend/requirements.txt --break-system-packages -q

# 2a. Install Whisper (audio/video transcription support)
echo "🎙️ Installing openai-whisper for audio/video support (this may take a few minutes)..."
pip install openai-whisper --break-system-packages -q

# 3. Launch backend
echo "🚀 Starting FastAPI backend on http://localhost:8000 ..."
cd backend
python main.py &
BACKEND_PID=$!
cd ..

# 4. Wait for backend to be ready
echo "⏳ Waiting for backend..."
for i in {1..15}; do
  sleep 1
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend ready!"
    break
  fi
done

echo ""
echo "══════════════════════════════════════════"
echo "  🧠 AI File Intelligence Bot is LIVE"
echo "  Backend API:  http://localhost:8000"
echo "  Frontend UI:  http://localhost:8000"
echo "  API Docs:     http://localhost:8000/docs"
echo "══════════════════════════════════════════"

wait $BACKEND_PID
