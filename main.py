import argparse
import os
import subprocess
import sys
import time
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


def run_dev():
    """Launch both FastAPI server and Streamlit UI concurrently."""
    print(f"⚡ Starting FastAPI Server (port {settings.fastapi_port}) AND Streamlit UI (port {settings.streamlit_port})...")
    
    # 1. Start FastAPI server as a subprocess
    api_cmd = [
        sys.executable, "-m", "uvicorn", "app.api.main:app",
        "--host", settings.fastapi_host,
        "--port", str(settings.fastapi_port)
    ]
    api_process = subprocess.Popen(api_cmd)
    
    time.sleep(2)  # Give API 2 seconds to initialize

    # 2. Start Streamlit UI
    ui_path = os.path.join(os.path.dirname(__file__), "ui", "app.py")
    ui_cmd = [
        sys.executable, "-m", "streamlit", "run", ui_path,
        "--server.port", str(settings.streamlit_port),
        "--server.headless", "true"
    ]
    
    try:
        subprocess.run(ui_cmd)
    finally:
        api_process.terminate()


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
        choices=["api", "ui", "dev", "test"],
        default="dev",
        help="Mode to execute: 'dev' (FastAPI + Streamlit together), 'api' (FastAPI Backend), 'ui' (Streamlit Interface), or 'test' (Pytest suite)"
    )

    args = parser.parse_args()

    if args.mode == "dev":
        run_dev()
    elif args.mode == "api":
        run_api()
    elif args.mode == "ui":
        run_ui()
    elif args.mode == "test":
        run_tests()


if __name__ == "__main__":
    main()
