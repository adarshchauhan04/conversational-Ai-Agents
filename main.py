import argparse
import os
import subprocess
import sys
import uvicorn
from app.config import settings


def run_api():
    """Launch the FastAPI server."""
    print(f"🚀 Starting FastAPI server on http://{settings.fastapi_host}:{settings.fastapi_port}...")
    uvicorn.run(
        "app.api.main:app",
        host=settings.fastapi_host,
        port=settings.fastapi_port,
        reload=True
    )


def run_ui():
    """Launch the Streamlit UI."""
    print(f"🎨 Starting Streamlit UI on http://localhost:{settings.streamlit_port}...")
    ui_path = os.path.join(os.path.dirname(__file__), "ui", "app.py")
    cmd = [
        sys.executable, "-m", "streamlit", "run", ui_path,
        "--server.port", str(settings.streamlit_port),
        "--server.headless", "true"
    ]
    subprocess.run(cmd)


def run_tests():
    """Run pytest suite."""
    print("🧪 Running Pytest Test Suite...")
    cmd = [sys.executable, "-m", "pytest", "-v"]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(description="Conversational AI Agent CLI Runner")
    parser.add_argument(
        "--mode",
        choices=["api", "ui", "test"],
        default="api",
        help="Mode to execute: 'api' (FastAPI Backend), 'ui' (Streamlit Interface), or 'test' (Pytest suite)"
    )

    args = parser.parse_args()

    if args.mode == "api":
        run_api()
    elif args.mode == "ui":
        run_ui()
    elif args.mode == "test":
        run_tests()


if __name__ == "__main__":
    main()
