#!/usr/bin/env python3
"""
PRD-AGI Phase 6 + Fuzzy Logic — Smart Launcher
Starts Streamlit and shows local + WiFi URLs.
"""
import subprocess
import socket
import sys
import os
import time
import threading
import webbrowser

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def open_browser(port: int, delay: float = 2.5):
    time.sleep(delay)
    webbrowser.open(f"http://localhost:{port}")

def main():
    port = int(os.getenv("PORT", "8501"))
    local_ip = get_local_ip()

    banner = f"""
{'='*62}
  🌀  PRD-AGI Phase 6 + Fuzzy Logic
      The Nameless Intelligence
{'='*62}

  💻  Local :  http://localhost:{port}
  📱  Phone :  http://{local_ip}:{port}

  → Both devices must be on the same WiFi network
  → Press Ctrl+C to stop
{'='*62}
"""
    print(banner)

    # Open browser automatically after 2.5s
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    cmd = [
        sys.executable, "-m", "streamlit", "run", "main.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n  👋  PRD-AGI stopped.\n")

if __name__ == "__main__":
    main()
