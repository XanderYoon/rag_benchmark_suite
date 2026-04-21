from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MCP_PROCESS_PID_KEY = "mcp_server_pid"
MCP_PROCESS_HOST_KEY = "mcp_server_host"
MCP_PROCESS_PORT_KEY = "mcp_server_port"
MCP_PROCESS_STARTED_AT_KEY = "mcp_server_started_at"
MCP_PROCESS_LOG_PATH_KEY = "mcp_server_log_path"


def render(show_title: bool = True) -> None:
    """Render the MCP control panel for hosting the local inspection server."""
    if show_title:
        st.title("MCP")
    else:
        st.subheader("MCP")
    st.subheader("MCP Server")
    st.caption("Host a local MCP endpoint for `retrieve_evidence` and `run_retrieval_benchmark`.")

    host = st.text_input(
        "Host",
        value=str(st.session_state.get(MCP_PROCESS_HOST_KEY, DEFAULT_HOST)),
        key="mcp_host_input",
        help="Network interface that the MCP HTTP server should bind to.",
    )
    port = st.number_input(
        "Port",
        min_value=1024,
        max_value=65535,
        value=int(st.session_state.get(MCP_PROCESS_PORT_KEY, DEFAULT_PORT)),
        step=1,
        key="mcp_port_input",
        help="Port for the MCP HTTP server endpoint.",
    )

    status = _server_status()
    if status["running"]:
        st.success(f"MCP server is running on {status['endpoint']}")
    else:
        st.info("MCP server is not running.")

    action_col_start, action_col_stop = st.columns(2)
    if action_col_start.button("Start MCP Server", key="start_mcp_server"):
        ok, message = _start_mcp_server(host=str(host).strip(), port=int(port))
        if ok:
            st.success(message)
        else:
            st.error(message)
        st.rerun()
    if action_col_stop.button("Stop MCP Server", key="stop_mcp_server"):
        ok, message = _stop_mcp_server()
        if ok:
            st.success(message)
        else:
            st.warning(message)
        st.rerun()

    if status["running"]:
        st.code(status["endpoint"], language="text")
        st.caption(f"Health check: {status['health_url']}")
        if status["log_path"]:
            st.caption(f"Server log: {status['log_path']}")
        st.markdown("**MCP Inspector**")
        st.write("Choose `Streamable HTTP` in MCP Inspector and use the endpoint above as the server URL.")
    else:
        st.caption("Start the server to inspect it from MCP Inspector.")

    with st.expander("Available tools", expanded=True):
        st.markdown("`retrieve_evidence`")
        st.caption("Retrieve top chunks from a validated knowledge base with source metadata and scores.")
        st.markdown("`run_retrieval_benchmark`")
        st.caption("Run one benchmark configuration and return source-separated metrics and timing.")


def _start_mcp_server(*, host: str, port: int) -> tuple[bool, str]:
    """Start the MCP server in a detached subprocess and verify readiness."""
    running_status = _server_status()
    if running_status["running"]:
        return True, f"MCP server is already running on {running_status['endpoint']}."

    command = [
        sys.executable,
        "-m",
        "rag_benchmark_mcp.server",
        "--host",
        host,
        "--port",
        str(port),
    ]
    log_path = Path("/tmp") / f"rag_benchmark_mcp_{port}.log"
    try:
        log_file = log_path.open("ab")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as exc:
        return False, f"Failed to start MCP server: {exc}"

    st.session_state[MCP_PROCESS_PID_KEY] = int(process.pid)
    st.session_state[MCP_PROCESS_HOST_KEY] = host
    st.session_state[MCP_PROCESS_PORT_KEY] = int(port)
    st.session_state[MCP_PROCESS_STARTED_AT_KEY] = time.time()
    st.session_state[MCP_PROCESS_LOG_PATH_KEY] = str(log_path)

    health_url = _health_url(host=host, port=port)
    for _ in range(10):
        if _is_mcp_server_ready(health_url=health_url):
            return True, f"Started MCP server on {_endpoint(host=host, port=port)}."
        time.sleep(0.3)
    return False, f"Started MCP process but could not verify readiness at {health_url}. Check log: {log_path}"


def _stop_mcp_server() -> tuple[bool, str]:
    """Stop the detached MCP subprocess when one is tracked in session state."""
    pid = st.session_state.get(MCP_PROCESS_PID_KEY)
    if not isinstance(pid, int):
        return False, "No tracked MCP server process is running in this session."

    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        _clear_mcp_session_state()
        return False, "Tracked MCP server process no longer exists."
    except Exception as exc:
        return False, f"Failed to stop MCP server: {exc}"

    _clear_mcp_session_state()
    return True, "Stopped MCP server."


def _server_status() -> dict[str, object]:
    """Return the tracked MCP server status for the current Streamlit session."""
    pid = st.session_state.get(MCP_PROCESS_PID_KEY)
    host = str(st.session_state.get(MCP_PROCESS_HOST_KEY, DEFAULT_HOST))
    port = int(st.session_state.get(MCP_PROCESS_PORT_KEY, DEFAULT_PORT))
    log_path = st.session_state.get(MCP_PROCESS_LOG_PATH_KEY)
    endpoint = _endpoint(host=host, port=port)
    health_url = _health_url(host=host, port=port)
    running = isinstance(pid, int) and _process_exists(pid) and _is_mcp_server_ready(health_url=health_url)
    if not running and isinstance(pid, int) and not _process_exists(pid):
        _clear_mcp_session_state()
    return {
        "running": running,
        "pid": pid,
        "endpoint": endpoint,
        "health_url": health_url,
        "log_path": log_path,
    }


def _is_mcp_server_ready(*, health_url: str) -> bool:
    """Return True when the MCP health endpoint responds successfully."""
    try:
        response = requests.get(health_url, timeout=1.5)
    except requests.RequestException:
        return False
    return response.status_code == 200


def _process_exists(pid: int) -> bool:
    """Return True when a process id still exists."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _endpoint(*, host: str, port: int) -> str:
    """Return the MCP endpoint URL for the configured host and port."""
    return f"http://{host}:{port}/mcp"


def _health_url(*, host: str, port: int) -> str:
    """Return the MCP health endpoint URL for the configured host and port."""
    return f"http://{host}:{port}/healthz"


def _clear_mcp_session_state() -> None:
    """Remove tracked MCP server metadata from the current Streamlit session."""
    st.session_state.pop(MCP_PROCESS_PID_KEY, None)
    st.session_state.pop(MCP_PROCESS_HOST_KEY, None)
    st.session_state.pop(MCP_PROCESS_PORT_KEY, None)
    st.session_state.pop(MCP_PROCESS_STARTED_AT_KEY, None)
    st.session_state.pop(MCP_PROCESS_LOG_PATH_KEY, None)
