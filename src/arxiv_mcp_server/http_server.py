"""
HTTP Server wrapper for ArXiv MCP Server
This allows the MCP server to be accessed via HTTP/HTTPS for Railway deployment
"""
import json
import logging
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .config import Settings
from .tools import handle_search, handle_download, handle_list_papers, handle_read_paper
from .tools import search_tool, download_tool, list_tool, read_tool
from .prompts.handlers import list_prompts as handler_list_prompts
from .prompts.handlers import get_prompt as handler_get_prompt

settings = Settings()
logger = logging.getLogger("arxiv-mcp-server")
logger.setLevel(logging.INFO)

# Create FastAPI app
app = FastAPI(title="ArXiv MCP Server", version=settings.APP_VERSION)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key authentication (required for security)
API_KEY = os.environ.get("MCP_API_KEY")  # Set in Railway environment variables

# Log warning if API key is not set (for production security)
if not API_KEY:
    logger.warning("⚠️  MCP_API_KEY not set! The server will accept all requests without authentication.")
    logger.warning("⚠️  For production deployment, please set MCP_API_KEY environment variable.")


def verify_api_key(request: Request) -> bool:
    """Verify API key from request headers.
    
    Supports two header formats:
    1. Authorization: Bearer <api_key>
    2. X-API-Key: <api_key>
    
    Returns:
        bool: True if API key is valid or not required, False otherwise
    """
    # If no API key is set, allow all requests (backward compatible, but not recommended for production)
    if not API_KEY:
        return True
    
    # Check for API key in headers
    # Poke sends it as Authorization: Bearer <key> or X-API-Key header
    auth_header = request.headers.get("Authorization", "")
    api_key_header = request.headers.get("X-API-Key", "")
    
    # Extract token from Bearer format
    token = None
    if auth_header.startswith("Bearer "):
        token = auth_header.replace("Bearer ", "").strip()
    elif api_key_header:
        token = api_key_header.strip()
    
    if not token:
        logger.warning("API key authentication failed: No API key provided in request headers")
        return False
    
    # Use constant-time comparison to prevent timing attacks
    if len(token) != len(API_KEY):
        logger.warning("API key authentication failed: Invalid API key length")
        return False
    
    # Constant-time comparison
    result = 0
    for x, y in zip(token.encode(), API_KEY.encode()):
        result |= x ^ y
    
    is_valid = result == 0
    if not is_valid:
        logger.warning("API key authentication failed: Invalid API key")
    
    return is_valid


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": settings.APP_NAME, "version": settings.APP_VERSION}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


def get_tool_definitions():
    """List available arXiv tools."""
    return [search_tool, download_tool, list_tool, read_tool]


async def call_tool(name: str, arguments: dict):
    """Handle tool calls for arXiv API."""
    try:
        if name == "search_papers":
            result = await handle_search(arguments)
        elif name == "download_paper":
            result = await handle_download(arguments)
        elif name == "list_papers":
            result = await handle_list_papers(arguments)
        elif name == "read_paper":
            result = await handle_read_paper(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
        
        # The handlers return List[TextContent], extract the text from each
        # Return as a single string (concatenated) or as list of text strings
        if isinstance(result, list):
            texts = []
            for item in result:
                if hasattr(item, 'text'):
                    texts.append(item.text)
                elif hasattr(item, 'model_dump'):
                    # If it's a TextContent object, extract text
                    item_dict = item.model_dump()
                    if 'text' in item_dict:
                        texts.append(item_dict['text'])
                    else:
                        texts.append(json.dumps(item_dict))
                else:
                    texts.append(str(item))
            # Return concatenated text (tools typically return single TextContent)
            return "\n".join(texts) if texts else ""
        elif hasattr(result, 'model_dump'):
            return result.model_dump()
        else:
            return result
    except Exception as e:
        logger.error(f"Error calling tool {name}: {e}", exc_info=True)
        raise ValueError(f"Error processing arXiv query: {str(e)}")


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """MCP protocol endpoint - JSON-RPC 2.0"""
    # Verify API key if configured
    if API_KEY and not verify_api_key(request):
        return JSONResponse(
            status_code=401,
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32001,
                    "message": "Unauthorized: Invalid or missing API key"
                }
            }
        )
    
    try:
        body = await request.json()
        
        # Handle MCP protocol messages
        if body.get("jsonrpc") == "2.0":
            method = body.get("method")
            params = body.get("params", {})
            
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {},
                            "prompts": {}
                        },
                        "serverInfo": {
                            "name": settings.APP_NAME,
                            "version": settings.APP_VERSION
                        }
                    }
                }
            elif method == "tools/list":
                tools = get_tool_definitions()
                # Convert tools to dict format, removing null fields
                tools_list = []
                for tool in tools:
                    tool_dict = tool.model_dump()
                    # Remove null fields for MCP Inspector compatibility
                    if tool_dict.get("title") is None:
                        tool_dict.pop("title", None)
                    if tool_dict.get("icons") is None:
                        tool_dict.pop("icons", None)
                    if tool_dict.get("outputSchema") is None:
                        tool_dict.pop("outputSchema", None)
                    if tool_dict.get("annotations") is None:
                        tool_dict.pop("annotations", None)
                    if tool_dict.get("execution") is None:
                        tool_dict.pop("execution", None)
                    if tool_dict.get("meta") is None:
                        tool_dict.pop("meta", None)
                    tools_list.append(tool_dict)
                return {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {
                        "tools": tools_list
                    }
                }
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                result = await call_tool(tool_name, arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, default=str, indent=2) if not isinstance(result, str) else result
                            }
                        ]
                    }
                }
            elif method == "prompts/list":
                prompts = await handler_list_prompts()
                prompts_list = []
                for prompt in prompts:
                    prompt_dict = prompt.model_dump()
                    # Remove null fields
                    if prompt_dict.get("annotations") is None:
                        prompt_dict.pop("annotations", None)
                    prompts_list.append(prompt_dict)
                return {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {
                        "prompts": prompts_list
                    }
                }
            elif method == "prompts/get":
                prompt_name = params.get("name")
                prompt_args = params.get("arguments")
                result = await handler_get_prompt(prompt_name, prompt_args)
                return {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": result.model_dump()
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "error": {"code": -32601, "message": f"Method not found: {method}"}
                }
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid JSON-RPC request"}
            )
    except Exception as e:
        logger.error(f"Error processing MCP request: {e}", exc_info=True)
        return {
            "jsonrpc": "2.0",
            "id": body.get("id") if 'body' in locals() else None,
            "error": {"code": -32603, "message": str(e)}
        }


@app.get("/tools")
async def list_tools_endpoint(request: Request):
    """List all available tools (requires API key if configured)"""
    # Verify API key if configured
    if API_KEY and not verify_api_key(request):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing API key"
        )
    
    tools = get_tool_definitions()
    return {
        "tools": [tool.model_dump() for tool in tools]
    }


@app.get("/debug/api-key")
async def debug_api_key():
    """Debug endpoint to check if API_KEY is loaded (remove in production if needed)
    
    This endpoint shows whether API key is configured but does not reveal the key itself.
    """
    return {
        "api_key_configured": bool(API_KEY),
        "api_key_length": len(API_KEY) if API_KEY else 0,
        "message": "API key is configured" if API_KEY else "⚠️  API key is NOT configured - server is open to all requests"
    }


def run_http_server():
    """Run the HTTP server using uvicorn."""
    host = os.getenv("HOST", settings.HOST)
    port = int(os.getenv("PORT", settings.PORT))
    
    logger.info(f"Starting HTTP server on {host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )
