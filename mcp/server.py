from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from mcp.contracts import JsonRpcError, JsonRpcRequest, JsonRpcResponse, ToolCallParams
from mcp.tools import build_tool_definitions, call_tool


SERVER_NAME = "rag-benchmark-mcp"
SERVER_VERSION = "0.1.0"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_PROTOCOL_VERSION = "2024-11-05"


class McpHttpHandler(BaseHTTPRequestHandler):
    """Serve a minimal MCP-compatible JSON-RPC surface over HTTP."""

    server_version = "RagBenchmarkMcp/0.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._write_json(HTTPStatus.OK, {"status": "ok", "server": SERVER_NAME, "version": SERVER_VERSION})
            return
        if self.path in {"/", "/mcp"}:
            self._write_json(
                HTTPStatus.OK,
                {
                    "server": SERVER_NAME,
                    "version": SERVER_VERSION,
                    "endpoint": "/mcp",
                    "inspector_transport": "Streamable HTTP",
                },
            )
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path '{self.path}'."})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/mcp":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path '{self.path}'."})
            return

        try:
            payload = self._read_json_body()
            request = JsonRpcRequest.model_validate(payload)
        except Exception as exc:
            response = JsonRpcResponse(
                id=None,
                error=JsonRpcError(code=-32700, message="Invalid JSON-RPC request.", data=str(exc)),
            )
            self._write_json(HTTPStatus.BAD_REQUEST, response.model_dump(mode="json"))
            return

        response = _dispatch_request(request=request)
        if response is None:
            self.send_response(HTTPStatus.ACCEPTED)
            self.end_headers()
            return
        self._write_json(HTTPStatus.OK, response.model_dump(mode="json"))

    def log_message(self, format: str, *args: Any) -> None:
        """Silence default HTTP request logging for cleaner subprocess output."""
        return

    def _read_json_body(self) -> dict[str, Any]:
        """Read and parse one JSON request body."""
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header.") from exc
        if content_length <= 0:
            raise ValueError("Request body is required.")
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Failed to parse JSON request body.") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON-RPC request body must be an object.")
        return payload

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        """Write one JSON response body."""
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(*, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Run the MCP HTTP server until interrupted."""
    httpd = ThreadingHTTPServer((host, port), McpHttpHandler)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


def _dispatch_request(*, request: JsonRpcRequest) -> JsonRpcResponse | None:
    """Dispatch one JSON-RPC request to the supported MCP method surface."""
    if request.method == "notifications/initialized":
        return None
    if request.method == "initialize":
        protocol_version = _requested_protocol_version(request.params)
        return JsonRpcResponse(
            id=request.id,
            result={
                "protocolVersion": protocol_version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )
    if request.method == "ping":
        return JsonRpcResponse(id=request.id, result={})
    if request.method == "tools/list":
        return JsonRpcResponse(id=request.id, result={"tools": build_tool_definitions()})
    if request.method == "tools/call":
        return _handle_tool_call(request=request)
    return JsonRpcResponse(
        id=request.id,
        error=JsonRpcError(
            code=-32601,
            message=f"Method '{request.method}' is not supported by this MCP server.",
        ),
    )


def _handle_tool_call(*, request: JsonRpcRequest) -> JsonRpcResponse:
    """Execute one validated MCP tool call."""
    try:
        params = ToolCallParams.model_validate(request.params or {})
    except Exception as exc:
        return JsonRpcResponse(
            id=request.id,
            error=JsonRpcError(code=-32602, message="Invalid tools/call params.", data=str(exc)),
        )

    try:
        payload = call_tool(tool_name=params.name, arguments=params.arguments)
    except Exception as exc:
        return JsonRpcResponse(
            id=request.id,
            result={
                "content": [{"type": "text", "text": str(exc)}],
                "structuredContent": {"error": str(exc)},
                "isError": True,
            },
        )

    return JsonRpcResponse(
        id=request.id,
        result={
            "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
            "structuredContent": payload,
            "isError": False,
        },
    )


def _requested_protocol_version(params: dict[str, Any] | None) -> str:
    """Return the client-requested protocol version when one is provided."""
    if not isinstance(params, dict):
        return DEFAULT_PROTOCOL_VERSION
    requested = str(params.get("protocolVersion", "")).strip()
    return requested or DEFAULT_PROTOCOL_VERSION


def main() -> None:
    """Parse CLI arguments and run the MCP server."""
    parser = argparse.ArgumentParser(description="Run the RAG benchmark MCP server.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind host for the MCP HTTP server.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port for the MCP HTTP server.")
    args = parser.parse_args()
    run_server(host=str(args.host), port=int(args.port))


if __name__ == "__main__":
    main()
