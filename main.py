"""Main entrypoint for the AI Face Emotion Robot Controller application."""

import argparse
import sys
import threading
import time
import webbrowser
import uvicorn
from src.web.server import app, vision_system


def open_browser(url: str, delay: float = 1.2):
    """Wait for server to start, then open the default web browser."""
    time.sleep(delay)
    print(f"\n[+] Opening Cockpit Dashboard at {url}")
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(description="AI Face Emotion-Controlled Bot Cockpit")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web server host IP")
    parser.add_argument("--port", type=int, default=8000, help="Web server port")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--esp32-ip", type=str, default="192.168.4.1", help="Target ESP32 Wi-Fi IP")
    parser.add_argument("--esp32-port", type=int, default=4210, help="Target ESP32 UDP port")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open web browser")
    args = parser.parse_args()

    print("=====================================================================")
    print("      AI FACE EMOTION ROBOT CONTROLLER - MISSION COCKPIT            ")
    print("=====================================================================")
    print(f"  * Webcam Device:       Index {args.camera}")
    print(f"  * Web Dashboard:       http://{args.host}:{args.port}")
    print(f"  * ESP32 Wi-Fi Target:  {args.esp32_ip}:{args.esp32_port}")
    print("---------------------------------------------------------------------")
    print("  Controls Mapping (Configurable in Dashboard):")
    print("    [Happy]      -> FORWARD")
    print("    [Sad]        -> BACKWARD")
    print("    [Angry]      -> TURN LEFT")
    print("    [Surprised]  -> TURN RIGHT")
    print("    [Neutral]    -> STOP")
    print("=====================================================================")

    # Configure vision system
    vision_system.camera_index = args.camera
    vision_system.udp_tx.set_target(args.esp32_ip, args.esp32_port)

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except KeyboardInterrupt:
        print("\nShutting down AI Robot Controller...")
    finally:
        vision_system.stop()
        sys.exit(0)


if __name__ == "__main__":
    main()
