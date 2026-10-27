import http.server
import socketserver
import json
import os
import socket
import subprocess
import sys

PORT = 8080
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "portfolio_config.json")
UPDATE_SCRIPT = os.path.join(BASE_DIR, "update_market_data.py")


def get_lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class PortfolioHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/api/server_info"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            info = {
                "lan_ip": get_lan_ip(),
                "port": PORT,
                "mobile_url": f"http://{get_lan_ip()}:{PORT}"
            }
            self.wfile.write(json.dumps(info).encode("utf-8"))
            return
        super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/refresh"):
            try:
                res = subprocess.run(
                    [sys.executable, UPDATE_SCRIPT],
                    cwd=BASE_DIR,
                    capture_output=True,
                    text=True,
                    timeout=90
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "ok": res.returncode == 0,
                    "stdout": res.stdout,
                    "stderr": res.stderr
                }).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return

        if self.path.startswith("/api/save_config"):
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8")
                new_cfg = json.loads(body)
                with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                    json.dump(new_cfg, f, ensure_ascii=False, indent=2)
                # Run update_market_data.py to recompute everything
                subprocess.run([sys.executable, UPDATE_SCRIPT], cwd=BASE_DIR, timeout=90)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode("utf-8"))
            return


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    lan_ip = get_lan_ip()
    with socketserver.TCPServer(("0.0.0.0", PORT), PortfolioHandler) as httpd:
        print("=" * 60)
        print(f"Portfolio PWA Server running!")
        print(f"PC Browser:     http://localhost:{PORT}")
        print(f"Mobile Phone:   http://{lan_ip}:{PORT}")
        print("=" * 60)
        httpd.serve_forever()
