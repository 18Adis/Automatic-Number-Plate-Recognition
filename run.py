"""Start PlateSight on localhost. Run with --open-browser for desktop use."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import threading
import time
import urllib.request
import webbrowser


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PlateSight local application")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a port between 1024 and 65535.")
    os.chdir(Path(__file__).resolve().parent)
    url = f"http://127.0.0.1:{args.port}"
    if args.open_browser:
        def open_when_ready() -> None:
            for _ in range(60):
                try:
                    with urllib.request.urlopen(url + "/api/status", timeout=1):
                        webbrowser.open(url)
                        return
                except (OSError, urllib.error.URLError):
                    time.sleep(0.5)
        threading.Thread(target=open_when_ready, daemon=True).start()
    import uvicorn
    uvicorn.run("plate_app.main:app", host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
