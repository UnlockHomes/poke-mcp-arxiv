"""Setup script for arxiv-mcp-server.

This is a minimal setup.py for Railway deployment compatibility.
The main configuration is in pyproject.toml.
"""

from setuptools import setup, find_packages

setup(
    name="arxiv-mcp-server",
    version="0.3.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "arxiv>=2.1.0",
        "httpx>=0.24.0",
        "python-dateutil>=2.8.2",
        "pydantic>=2.8.0",
        "mcp>=1.2.0",
        "pymupdf4llm>=0.0.17",
        "aiohttp>=3.9.1",
        "python-dotenv>=1.0.0",
        "pydantic-settings>=2.1.0",
        "aiofiles>=23.2.1",
        "uvicorn>=0.30.0",
        "sse-starlette>=1.8.2",
        "fastapi>=0.104.0",
        "anyio>=4.2.0",
        "black>=25.1.0",
    ],
    python_requires=">=3.11",
)

