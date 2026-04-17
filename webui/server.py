#!/usr/bin/env python3
"""CSV 数据查看器 - 本地 HTTP 服务器

启动方式: python server.py
访问地址: http://localhost:8080
"""

import csv
import io
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "result"
WEBUI_DIR = Path(__file__).resolve().parent


class CSVViewerHandler(SimpleHTTPRequestHandler):
    """处理静态文件和 CSV 数据 API"""

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/files":
            self._handle_list_files()
        elif parsed.path == "/api/file":
            self._handle_read_file(parsed)
        else:
            self._serve_static(parsed.path)

    # ── API: 列出 data 目录下的 CSV 文件 ──────────────────────────

    def _handle_list_files(self):
        try:
            files = sorted(f.name for f in DATA_DIR.glob("*.csv"))
            self._send_json(files)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    # ── API: 读取指定 CSV 文件内容 ────────────────────────────────

    def _handle_read_file(self, parsed):
        params = parse_qs(parsed.query)
        name = params.get("name", [""])[0]

        # 安全校验
        if not name or ".." in name or "/" in name or "\\" in name:
            self._send_json({"error": "无效的文件名"}, 400)
            return

        file_path = DATA_DIR / name
        if not file_path.exists():
            self._send_json({"error": "文件不存在"}, 404)
            return

        try:
            content = self._read_file_content(file_path)
            reader = csv.reader(io.StringIO(content))
            rows = list(reader)

            if not rows:
                self._send_json({"headers": [], "rows": []})
                return

            self._send_json({"headers": rows[0], "rows": rows[1:]})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    @staticmethod
    def _read_file_content(file_path: Path) -> str:
        for encoding in ("utf-8-sig", "utf-8", "gbk"):
            try:
                return file_path.read_text(encoding=encoding)
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise ValueError(f"无法解码文件: {file_path.name}")

    # ── 静态文件服务 ──────────────────────────────────────────────

    def _serve_static(self, path: str):
        if path == "/":
            path = "/index.html"

        file_path = WEBUI_DIR / path.lstrip("/")
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404)
            return

        content_type = self.guess_type(str(file_path))
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.end_headers()
        with open(file_path, "rb") as f:
            self.wfile.write(f.read())

    # ── 工具方法 ──────────────────────────────────────────────────

    def _send_json(self, data, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # 静默日志


if __name__ == "__main__":
    port = 8380
    server = HTTPServer(("0.0.0.0", port), CSVViewerHandler)
    print(f"CSV 数据查看器已启动: http://0.0.0.0:{port}")
    print(f"数据目录: {DATA_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
