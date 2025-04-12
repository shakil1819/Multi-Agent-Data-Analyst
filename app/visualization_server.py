import os
import socket
import threading
import time
import webbrowser

from http.server import HTTPServer, SimpleHTTPRequestHandler


# Default port for serving visualizations
DEFAULT_PORT = 8081


class VisualizationServer:
    """A simple HTTP server for serving visualization HTML files."""

    def __init__(self, port=DEFAULT_PORT):
        self.port = port
        self.server = None
        self.thread = None
        self._is_running = False
        self._find_available_port()

    def _find_available_port(self):
        """Find an available port starting from the default."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port = self.port
        max_port = self.port + 100  # Try up to 100 ports

        while port < max_port:
            try:
                s.bind(("", port))
                self.port = port
                s.close()
                return
            except OSError:
                port += 1

        # If we get here, we couldn't find an available port
        self.port = self.port  # Use the original port and hope for the best
        s.close()

    def start(self, directory=".", open_browser=False, path=None):
        """Start the visualization server."""
        if self._is_running:
            return f"http://localhost:{self.port}"

        # Change to the appropriate directory
        os.chdir(directory)

        # Create and start the server
        handler = SimpleHTTPRequestHandler
        self.server = HTTPServer(("", self.port), handler)
        self.thread = threading.Thread(target=self._run_server)
        self.thread.daemon = True
        self.thread.start()
        self._is_running = True

        # Wait a bit to ensure server is up
        time.sleep(0.5)

        # Construct the URL
        url = f"http://localhost:{self.port}"

        if path:
            url = f"{url}/{os.path.basename(path)}"

        # Open in browser if requested
        if open_browser:
            webbrowser.open(url)

        return url

    def _run_server(self):
        """Run the server on the thread."""
        try:
            self.server.serve_forever()
        except Exception:
            self._is_running = False

    def stop(self):
        """Stop the server."""
        if self.server and self._is_running:
            self.server.shutdown()
            self._is_running = False


# Global server instance
_server = None


def get_server():
    """Get or create the visualization server instance."""
    global _server
    if _server is None:
        _server = VisualizationServer()
    return _server


def get_visualization_url(viz_path):
    """Get a URL to view the visualization file."""
    if not os.path.exists(viz_path):
        return None

    server = get_server()
    dir_path = os.path.dirname(os.path.abspath(viz_path))
    return server.start(directory=dir_path, path=viz_path)


def serve_visualization(viz_path, open_browser=False):
    """Serve a visualization file and return its URL."""
    if not os.path.exists(viz_path):
        return None

    server = get_server()
    dir_path = os.path.dirname(os.path.abspath(viz_path))
    return server.start(directory=dir_path, open_browser=open_browser, path=viz_path)
