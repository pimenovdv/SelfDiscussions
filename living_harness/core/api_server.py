import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from living_harness.core.agent import Agent

class AgentAPIHandler(BaseHTTPRequestHandler):
    agent: Optional[Agent] = None

    def do_POST(self):
        if self.path == '/chat':
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "Empty request"}')
                return

            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode('utf-8'))
                message = data.get('message', '')

                if not message:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'{"error": "Message is required"}')
                    return

                if self.agent:
                    result = self.agent.process_input(message)
                else:
                    result = {"response": f"Mock response to {message}"}

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))

            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "Invalid JSON"}')
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "Not Found"}')

def run_server(agent: Agent, host: str = 'localhost', port: int = 8000):
    AgentAPIHandler.agent = agent
    server = HTTPServer((host, port), AgentAPIHandler)
    print(f"Starting server on {host}:{port}...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.server_close()

if __name__ == "__main__":
    test_agent = Agent(name="TestAgent", system_prompt="You are a test agent.")
    run_server(test_agent, port=8001)
