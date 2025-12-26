"""
Arxiv MCP Server initialization
"""

import os
import asyncio
from . import server


def main():
    """Main entry point for the package.
    
    Supports two modes:
    - stdio: Default mode for local MCP clients (via MCP_TRANSPORT=stdio or unset)
    - http: HTTP/SSE mode for cloud deployment (via MCP_TRANSPORT=http)
    """
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    
    if transport == "http":
        # Run HTTP server
        from .http_server import run_http_server
        run_http_server()
    else:
        # Run stdio server (default)
        asyncio.run(server.main())


__all__ = ["main", "server"]
