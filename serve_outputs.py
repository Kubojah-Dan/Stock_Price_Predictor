import http.server
import socketserver
from pathlib import Path

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

PORT = 8000
DIRECTORY = "outputs"

if __name__ == "__main__":
    import os
    os.chdir(DIRECTORY)
    
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"Serving {DIRECTORY} at http://localhost:{PORT}")
        print("Press Ctrl+C to stop")
        httpd.serve_forever()
