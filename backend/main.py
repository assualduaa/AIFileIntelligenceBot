"""
main.py — Application entry point
Run: python main.py  OR  uvicorn main:app --reload
"""
import uvicorn
from api import app
from config import HOST, PORT

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info",
    )
