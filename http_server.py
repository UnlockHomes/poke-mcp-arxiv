"""
HTTP Server entry point for ArXiv MCP Server
This allows the MCP server to be accessed via HTTP/HTTPS for Railway deployment
"""
import sys
import os

# Set HTTP transport mode
os.environ["MCP_TRANSPORT"] = "http"

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the HTTP server
from arxiv_mcp_server.http_server import run_http_server

if __name__ == "__main__":
    run_http_server()

