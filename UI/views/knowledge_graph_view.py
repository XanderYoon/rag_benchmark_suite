from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

from RAG.services.graph_view_service import build_interactive_graph_payload, load_graph_view_data
from UI.state.session_state import get_loaded_knowledge_base


def _show_title(show_title: bool) -> None:
    if show_title:
        st.title("Knowledge-Base")
    else:
        st.subheader("Knowledge-Base")
    st.subheader("View Knowledge Graph")


def render(show_title: bool = True) -> None:
    _show_title(show_title)
    loaded_knowledge_base = get_loaded_knowledge_base()
    if loaded_knowledge_base is None:
        st.warning("No knowledge base is loaded. Load a LightRAG or GraphRAG knowledge base first.")
        return

    try:
        graph_payload = load_graph_view_data(knowledge_base=loaded_knowledge_base)
    except Exception as exc:
        st.warning(str(exc))
        return

    st.info(
        "Loaded graph-capable knowledge base: "
        f"{loaded_knowledge_base.get('knowledge_base_dir', '')} "
        f"({graph_payload.get('method_id', '')})"
    )
    st.caption(
        f"Nodes: {graph_payload.get('node_count', 0)} | "
        f"Edges: {graph_payload.get('edge_count', 0)}"
    )

    paper_ids = ["All papers"] + list(graph_payload.get("paper_ids", []))
    selected_paper_label = st.selectbox(
        "Paper filter",
        options=paper_ids,
        help="Filter the graph to one paper or view the highest-degree nodes across the loaded graph.",
    )
    max_nodes = int(
        st.slider(
            "Max nodes to render",
            min_value=5,
            max_value=min(150, max(5, int(graph_payload.get("node_count", 5)))),
            value=min(25, max(5, int(graph_payload.get("node_count", 25)))),
            step=5,
            help="Large graphs are filtered to keep the visualization readable.",
        )
    )

    selected_paper_id = None if selected_paper_label == "All papers" else selected_paper_label
    interactive_payload = build_interactive_graph_payload(
        graph_payload=graph_payload,
        selected_paper_id=selected_paper_id,
        max_nodes=max_nodes,
    )
    visible_nodes = list(interactive_payload.get("nodes", []))
    visible_edges = list(interactive_payload.get("edges", []))

    if not visible_nodes:
        st.info("No graph nodes matched the current filter.")
        return

    components.html(_build_interactive_graph_html(interactive_payload=interactive_payload), height=640)
    st.caption(
        "Pan by dragging the background. Zoom with the mouse wheel. Drag nodes to reposition them. "
        "Edge labels show graph weights."
    )
    st.caption(f"Showing {len(visible_nodes)} nodes and {len(visible_edges)} edges in the current view.")

    with st.expander("Visible nodes"):
        st.dataframe(
            [
                {
                    "paper_id": node.get("paper_id", ""),
                    "chunk_id": node.get("chunk_id", ""),
                    "degree": node.get("degree", 0),
                    "preview_text": node.get("preview_text", ""),
                }
                for node in visible_nodes
            ],
            use_container_width=True,
        )

    with st.expander("Visible edges"):
        st.dataframe(visible_edges, use_container_width=True)


def _build_interactive_graph_html(*, interactive_payload: dict) -> str:
    """Build an interactive SVG graph with pan, zoom, drag, and hover tooltips."""
    payload_json = json.dumps(interactive_payload)
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    body {{
      margin: 0;
      font-family: sans-serif;
      background: #f8fafc;
    }}
    .graph-shell {{
      position: relative;
      width: 100%;
      height: 620px;
      border: 1px solid #cbd5e1;
      border-radius: 12px;
      overflow: hidden;
      background:
        radial-gradient(circle at top left, rgba(125, 211, 252, 0.20), transparent 28%),
        linear-gradient(180deg, #f8fafc 0%, #eef6ff 100%);
    }}
    svg {{
      width: 100%;
      height: 100%;
      display: block;
      cursor: grab;
    }}
    svg.dragging {{
      cursor: grabbing;
    }}
    .edge {{
      stroke: #94a3b8;
      stroke-opacity: 0.85;
    }}
    .edge-label {{
      font-size: 11px;
      fill: #334155;
      text-anchor: middle;
      paint-order: stroke;
      stroke: rgba(248, 250, 252, 0.96);
      stroke-width: 4px;
      stroke-linejoin: round;
      pointer-events: none;
    }}
    .node {{
      fill: #e0f2fe;
      stroke: #0284c7;
      stroke-width: 2px;
      cursor: pointer;
    }}
    .node-label {{
      font-size: 11px;
      fill: #0f172a;
      pointer-events: none;
      text-anchor: middle;
    }}
    .tooltip {{
      position: absolute;
      pointer-events: none;
      max-width: 280px;
      padding: 10px 12px;
      border-radius: 10px;
      background: rgba(15, 23, 42, 0.92);
      color: #f8fafc;
      font-size: 12px;
      line-height: 1.4;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.20);
      opacity: 0;
      transform: translate(12px, 12px);
      transition: opacity 0.12s ease;
      white-space: pre-wrap;
    }}
    .tooltip.visible {{
      opacity: 1;
    }}
    .legend {{
      position: absolute;
      top: 10px;
      right: 12px;
      padding: 8px 10px;
      border-radius: 10px;
      background: rgba(255, 255, 255, 0.85);
      border: 1px solid #cbd5e1;
      font-size: 12px;
      color: #334155;
    }}
  </style>
</head>
<body>
  <div class="graph-shell">
    <div class="legend">Hover for chunk preview</div>
    <svg id="graph" viewBox="0 0 1200 700" preserveAspectRatio="xMidYMid meet">
      <g id="viewport"></g>
    </svg>
    <div id="tooltip" class="tooltip"></div>
  </div>
  <script>
    const payload = {payload_json};
    const svg = document.getElementById("graph");
    const viewport = document.getElementById("viewport");
    const tooltip = document.getElementById("tooltip");
    const state = {{
      scale: 1,
      offsetX: 60,
      offsetY: 40,
      panActive: false,
      dragNodeId: null,
      lastX: 0,
      lastY: 0,
      nodes: (payload.nodes || []).map((node) => ({{ ...node }})),
      edges: payload.edges || [],
    }};
    const nodeIndex = new Map();
    state.nodes.forEach((node) => {{
      node.vx = 0;
      node.vy = 0;
      node.fx = null;
      node.fy = null;
      nodeIndex.set(node.chunk_id, node);
    }});

    function applyTransform() {{
      viewport.setAttribute("transform", `translate(${{state.offsetX}} ${{state.offsetY}}) scale(${{state.scale}})`);
    }}

    function escapeHtml(value) {{
      return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }}

    function showTooltip(event, node) {{
      tooltip.innerHTML = `<strong>${{escapeHtml(node.chunk_id)}}</strong><br>${{escapeHtml(node.preview_text || "No preview available.")}}`;
      tooltip.style.left = `${{event.offsetX + 8}}px`;
      tooltip.style.top = `${{event.offsetY + 8}}px`;
      tooltip.classList.add("visible");
    }}

    function hideTooltip() {{
      tooltip.classList.remove("visible");
    }}

    function graphPointFromClient(clientX, clientY) {{
      const rect = svg.getBoundingClientRect();
      return {{
        x: (clientX - rect.left - state.offsetX) / state.scale,
        y: (clientY - rect.top - state.offsetY) / state.scale,
      }};
    }}

    function runForceLayout(iterations = 220) {{
      const repulsion = 18000;
      const springLength = 150;
      const springStrength = 0.008;
      const centering = 0.0018;

      for (let step = 0; step < iterations; step += 1) {{
        for (const node of state.nodes) {{
          node.vx *= 0.86;
          node.vy *= 0.86;
        }}

        for (let i = 0; i < state.nodes.length; i += 1) {{
          const a = state.nodes[i];
          for (let j = i + 1; j < state.nodes.length; j += 1) {{
            const b = state.nodes[j];
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            let distSq = (dx * dx) + (dy * dy);
            if (distSq < 0.01) {{
              dx = 0.1;
              dy = 0.1;
              distSq = 0.02;
            }}
            const dist = Math.sqrt(distSq);
            const force = repulsion / distSq;
            const fx = (force * dx) / dist;
            const fy = (force * dy) / dist;
            a.vx -= fx;
            a.vy -= fy;
            b.vx += fx;
            b.vy += fy;
          }}
        }}

        for (const edge of state.edges) {{
          const source = nodeIndex.get(edge.source_chunk_id);
          const target = nodeIndex.get(edge.target_chunk_id);
          if (!source || !target) {{
            continue;
          }}
          let dx = target.x - source.x;
          let dy = target.y - source.y;
          let dist = Math.sqrt((dx * dx) + (dy * dy)) || 0.001;
          const desired = Math.max(80, springLength - (Number(edge.weight || 0) * 18));
          const force = (dist - desired) * springStrength;
          const fx = (force * dx) / dist;
          const fy = (force * dy) / dist;
          source.vx += fx;
          source.vy += fy;
          target.vx -= fx;
          target.vy -= fy;
        }}

        for (const node of state.nodes) {{
          const centerDx = 520 - node.x;
          const centerDy = 320 - node.y;
          node.vx += centerDx * centering;
          node.vy += centerDy * centering;
          if (node.fx !== null && node.fy !== null) {{
            node.x = node.fx;
            node.y = node.fy;
            node.vx = 0;
            node.vy = 0;
          }} else {{
            node.x += node.vx;
            node.y += node.vy;
          }}
        }}
      }}
    }}

    function render() {{
      viewport.innerHTML = "";

      for (const edge of state.edges) {{
        const source = nodeIndex.get(edge.source_chunk_id);
        const target = nodeIndex.get(edge.target_chunk_id);
        if (!source || !target) {{
          continue;
        }}
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", source.x);
        line.setAttribute("y1", source.y);
        line.setAttribute("x2", target.x);
        line.setAttribute("y2", target.y);
        line.setAttribute("stroke-width", Math.max(1.2, Math.min(4.5, 1 + Number(edge.weight || 0))));
        line.setAttribute("class", "edge");
        viewport.appendChild(line);

        const edgeLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
        edgeLabel.setAttribute("x", (source.x + target.x) / 2);
        edgeLabel.setAttribute("y", ((source.y + target.y) / 2) - 6);
        edgeLabel.setAttribute("class", "edge-label");
        edgeLabel.textContent = Number(edge.weight || 0).toFixed(2);
        viewport.appendChild(edgeLabel);
      }}

      for (const node of state.nodes) {{
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        group.dataset.chunkId = node.chunk_id;

        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", node.x);
        circle.setAttribute("cy", node.y);
        circle.setAttribute("r", node.radius || 20);
        circle.setAttribute("class", "node");
        circle.addEventListener("mouseenter", (event) => showTooltip(event, node));
        circle.addEventListener("mousemove", (event) => showTooltip(event, node));
        circle.addEventListener("mouseleave", hideTooltip);
        circle.addEventListener("mousedown", (event) => {{
          event.stopPropagation();
          state.dragNodeId = node.chunk_id;
          node.fx = node.x;
          node.fy = node.y;
          state.lastX = event.clientX;
          state.lastY = event.clientY;
        }});
        group.appendChild(circle);

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", node.x);
        label.setAttribute("y", node.y + 4);
        label.setAttribute("class", "node-label");
        label.textContent = String(node.paper_id || node.chunk_id).slice(0, 18);
        group.appendChild(label);

        viewport.appendChild(group);
      }}
      applyTransform();
    }}

    svg.addEventListener("mousedown", (event) => {{
      if (state.dragNodeId) {{
        return;
      }}
      state.panActive = true;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      svg.classList.add("dragging");
    }});

    window.addEventListener("mousemove", (event) => {{
      if (state.dragNodeId) {{
        const node = state.nodes.find((item) => item.chunk_id === state.dragNodeId);
        if (!node) {{
          return;
        }}
        const from = graphPointFromClient(state.lastX, state.lastY);
        const to = graphPointFromClient(event.clientX, event.clientY);
        node.x += (to.x - from.x);
        node.y += (to.y - from.y);
        node.fx = node.x;
        node.fy = node.y;
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        render();
        return;
      }}
      if (!state.panActive) {{
        return;
      }}
      state.offsetX += event.clientX - state.lastX;
      state.offsetY += event.clientY - state.lastY;
      state.lastX = event.clientX;
      state.lastY = event.clientY;
      applyTransform();
    }});

    window.addEventListener("mouseup", () => {{
      state.panActive = false;
      state.dragNodeId = null;
      for (const node of state.nodes) {{
        node.fx = null;
        node.fy = null;
      }}
      svg.classList.remove("dragging");
    }});

    svg.addEventListener("wheel", (event) => {{
      event.preventDefault();
      const delta = event.deltaY < 0 ? 1.1 : 0.9;
      const nextScale = Math.max(0.35, Math.min(3.5, state.scale * delta));
      const rect = svg.getBoundingClientRect();
      const cursorX = event.clientX - rect.left;
      const cursorY = event.clientY - rect.top;
      state.offsetX = cursorX - ((cursorX - state.offsetX) * (nextScale / state.scale));
      state.offsetY = cursorY - ((cursorY - state.offsetY) * (nextScale / state.scale));
      state.scale = nextScale;
      applyTransform();
    }}, {{ passive: false }});

    runForceLayout();
    render();
  </script>
</body>
</html>
"""
