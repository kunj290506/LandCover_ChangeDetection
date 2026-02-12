#!/usr/bin/env python3
"""
Frontend Server for Land Cover Change Detection
Serves the web interface and acts as proxy to backend API
"""

import os
import http.server
import socketserver
from pathlib import Path
import json
import urllib.parse
import urllib.request
from datetime import datetime

class FrontendHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent / "frontend"), **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle preflight requests"""
        self.send_response(200)
        self.end_headers()
    
    def do_POST(self):
        """Handle POST requests - proxy to backend API"""
        if self.path.startswith('/api/'):
            self.proxy_to_backend()
        else:
            self.send_error(404)
    
    def proxy_to_backend(self):
        """Proxy API requests to backend server"""
        try:
            # Read request data
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            # Forward to backend
            backend_url = f"http://localhost:8080{self.path.replace('/api', '')}"
            
            req = urllib.request.Request(
                backend_url,
                data=post_data,
                headers=dict(self.headers)
            )
            
            with urllib.request.urlopen(req) as response:
                self.send_response(response.status)
                
                # Forward response headers
                for header, value in response.headers.items():
                    if header.lower() not in ['connection', 'transfer-encoding']:
                        self.send_header(header, value)
                
                self.end_headers()
                
                # Forward response data
                self.wfile.write(response.read())
                
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            error_response = {
                "error": f"Proxy error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            self.wfile.write(json.dumps(error_response).encode())

def run_frontend_server(port=3000):
    """Run the frontend server"""
    try:
        with socketserver.TCPServer(("", port), FrontendHandler) as httpd:
            print("=" * 60)
            print("🎨 LAND COVER CHANGE DETECTION - FRONTEND SERVER")
            print("=" * 60)
            print(f"🌐 Frontend URL: http://localhost:{port}")
            print(f"📱 Open in browser: http://localhost:{port}")
            print(f"🔗 Backend API: http://localhost:8080")
            print(f"📁 Serving: frontend/ directory")
            print()
            print("✨ Features:")
            print("   • Drag & drop image upload")
            print("   • Real-time change detection")
            print("   • Interactive results visualization")
            print("   • Responsive design")
            print()
            print("⏹️  Press Ctrl+C to stop")
            print("=" * 60)
            
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n⏹️  Frontend server stopped")
    except Exception as e:
        print(f"❌ Frontend server error: {e}")

if __name__ == "__main__":
    run_frontend_server()