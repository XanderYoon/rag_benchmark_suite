from __future__ import annotations

import argparse
import contextlib
from collections.abc import AsyncIterator
from typing import Literal

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from rag_benchmark_mcp.contracts import (
    RetrieveEvidenceRequest,
    RetrieveEvidenceResult,
    RunRetrievalBenchmarkRequest,
    RunRetrievalBenchmarkResult,
)
from rag_benchmark_mcp.tools import retrieve_evidence, run_retrieval_benchmark


SERVER_NAME = "rag-benchmark-mcp"
SERVER_VERSION = "0.2.0"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

_mcp_server = FastMCP(
    SERVER_NAME,
    stateless_http=True,
    json_response=True,
)


@_mcp_server.tool(
    name="retrieve_evidence",
    description="Retrieve the highest-ranked evidence chunks from a validated knowledge base.",
)
def retrieve_evidence_tool(
    query: str,
    knowledge_base_path: str,
    retrieval_method: Literal["faiss", "graphrag", "lightrag"] | None = None,
    retrieval_model: str | None = None,
    retrieval_provider: Literal["openai", "ollama"] | None = None,
    top_k: int = 5,
) -> RetrieveEvidenceResult:
    """Retrieve top-ranked evidence chunks for one query."""
    return retrieve_evidence(
        request=RetrieveEvidenceRequest(
            query=query,
            knowledge_base_path=knowledge_base_path,
            retrieval_method=retrieval_method,
            retrieval_model=retrieval_model,
            retrieval_provider=retrieval_provider,
            top_k=top_k,
        )
    )


@_mcp_server.tool(
    name="run_retrieval_benchmark",
    description="Run one retrieval benchmark configuration and return normalized metrics.",
)
def run_retrieval_benchmark_tool(
    embedded_chunks_path: str,
    retrieval_model: str,
    evaluation_model: str | None = None,
    max_cases: int = 24,
    top_k: int = 5,
    tools: list[Literal["ragas"]] | None = None,
    include_auto_probes: bool = True,
    include_verified_probes: bool = False,
    verified_questions_path: str = "data/verified_questions.json",
    retrieval_methods: list[Literal["faiss", "graphrag", "lightrag"]] | None = None,
    telemetry_output_dir: str | None = None,
) -> RunRetrievalBenchmarkResult:
    """Execute one retrieval benchmark run."""
    return run_retrieval_benchmark(
        request=RunRetrievalBenchmarkRequest(
            embedded_chunks_path=embedded_chunks_path,
            retrieval_model=retrieval_model,
            evaluation_model=evaluation_model,
            max_cases=max_cases,
            top_k=top_k,
            tools=list(tools or ["ragas"]),
            include_auto_probes=include_auto_probes,
            include_verified_probes=include_verified_probes,
            verified_questions_path=verified_questions_path,
            retrieval_methods=list(retrieval_methods or ["faiss"]),
            telemetry_output_dir=telemetry_output_dir,
        )
    )


def build_mcp_server() -> FastMCP:
    """Return the configured FastMCP server instance."""
    return _mcp_server


async def _healthcheck(_: Request) -> JSONResponse:
    """Return a simple process health payload."""
    return JSONResponse({"status": "ok", "server": SERVER_NAME, "version": SERVER_VERSION})


async def _server_info(request: Request) -> JSONResponse:
    """Return connection hints for local inspection clients."""
    return JSONResponse(
        {
            "server": SERVER_NAME,
            "version": SERVER_VERSION,
            "endpoint": str(request.url.replace(path="/mcp", query="", fragment="")),
            "inspector_transport": "Streamable HTTP",
        }
    )


@contextlib.asynccontextmanager
async def _lifespan(_: Starlette) -> AsyncIterator[None]:
    """Run the FastMCP session manager for the mounted ASGI app."""
    async with _mcp_server.session_manager.run():
        yield


def build_http_app() -> CORSMiddleware:
    """Build the Starlette app that hosts the MCP transport and health routes."""
    app = Starlette(
        routes=[
            Route("/", endpoint=_server_info),
            Route("/healthz", endpoint=_healthcheck),
            Mount("/", app=_mcp_server.streamable_http_app()),
        ],
        lifespan=_lifespan,
    )
    return CORSMiddleware(
        app,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Mcp-Session-Id"],
    )


def run_server(*, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Run the MCP Streamable HTTP server until interrupted."""
    uvicorn.run(build_http_app(), host=host, port=port, log_level="warning")


def main() -> None:
    """Parse CLI arguments and run the MCP server."""
    parser = argparse.ArgumentParser(description="Run the RAG benchmark MCP server.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind host for the MCP HTTP server.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port for the MCP HTTP server.")
    args = parser.parse_args()
    run_server(host=str(args.host), port=int(args.port))


if __name__ == "__main__":
    main()
