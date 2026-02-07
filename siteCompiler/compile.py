#!/usr/bin/env python3
"""
Site compiler — reads GraphML graph data and generates a flat HTML website
with interactive Sigma.js visualization, search, and pathfinding.

Usage:
    python siteCompiler/compile.py -i output/ -o www/

The generated site works via file:// protocol (no server required).
All data is embedded as JS globals loaded via <script src> tags.
"""

import argparse
import json
import shutil
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

import networkx as nx
import numpy as np
from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = Path(__file__).parent / "templates"

# CDN URLs for JS libraries
JS_LIBS = {
    "graphology.umd.min.js": "https://unpkg.com/graphology@0.25.4/dist/graphology.umd.min.js",
    "sigma.min.js": "https://unpkg.com/sigma@2.4.0/build/sigma.min.js",
}

THEME = {
    "background": "#eef1f4",
    "edge_default": "rgba(72,82,96,0.22)",
    "edge_hover": "rgba(72,82,96,0.6)",
    "edge_dim": "rgba(72,82,96,0.08)",
    "edge_focus": "rgba(72,82,96,0.55)",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cluster_graph(input_dir: Path) -> nx.Graph:
    """Load the cluster-level graph."""
    path = input_dir / "cluster_graph.graphml"
    return nx.read_graphml(str(path))


def load_cluster_subgraphs(input_dir: Path) -> dict[int, nx.Graph]:
    """Load all per-cluster subgraphs."""
    graphs = {}
    for f in sorted(input_dir.glob("cluster_*.graphml")):
        if f.stem == "cluster_graph":
            continue
        cid = int(f.stem.split("_")[1])
        graphs[cid] = nx.read_graphml(str(f))
    return graphs


def load_names(input_dir: Path) -> dict[int, str]:
    """Load cluster names from names.json."""
    path = input_dir / "names.json"
    if path.exists():
        with open(path) as f:
            raw = json.load(f)
        return {int(k): str(v) for k, v in raw.items()}
    return {}


def load_full_graph(input_dir: Path) -> dict | None:
    """Load full_graph.json for pathfinding data."""
    path = input_dir / "full_graph.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_full_layout(input_dir: Path) -> dict[str, list[float]] | None:
    """Load full_layout.json with precomputed node positions from graph.py."""
    path = input_dir / "full_layout.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Layout & color computation
# ---------------------------------------------------------------------------

def compute_layout(g: nx.Graph, **kwargs) -> dict:
    """Compute spring layout and normalize positions to [0, 1]."""
    if g.number_of_nodes() == 0:
        return {}
    pos = nx.spring_layout(g, **kwargs)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    rx = max_x - min_x if max_x != min_x else 1.0
    ry = max_y - min_y if max_y != min_y else 1.0
    return {
        n: ((p[0] - min_x) / rx, (p[1] - min_y) / ry)
        for n, p in pos.items()
    }


def color_by_distance_from_center(pos: dict) -> dict[str, str]:
    """Rank-normalize distance from layout center → RdYlGn colormap hex."""
    if not pos:
        return {}
    nodes = list(pos.keys())
    coords = np.array([pos[n] for n in nodes])
    center = coords.mean(axis=0)
    dists = np.linalg.norm(coords - center, axis=1)
    n = len(dists)
    ranks = dists.argsort().argsort()
    # Invert: close to center = 1.0 (green), far = 0.0 (red)
    norm = np.array([1.0 - ranks[i] / max(n - 1, 1) for i in range(n)])
    colormap = cm.RdYlGn
    colors = {}
    for i, node in enumerate(nodes):
        rgba = colormap(norm[i])
        colors[node] = "#{:02x}{:02x}{:02x}".format(
            int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        )
    return colors


def rgba_to_hex(r, g, b) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def align_layout_to_clusters(
    full_layout: dict[str, list[float]],
    subgraphs: dict[int, nx.Graph],
    cg_pos: dict,
) -> dict[str, list[float]]:
    """Align the full graph layout so cluster centroids match the overview.

    Computes Procrustes (rotation + uniform scale + translation) from the DrL
    cluster centroids to the cluster overview positions, then applies that
    transform to every node. Result is re-normalized to [0, 1].
    """
    # Compute cluster centroids in the DrL layout
    cluster_ids = sorted(subgraphs.keys())
    drl_centroids = []
    target_centroids = []
    for cid in cluster_ids:
        sg = subgraphs[cid]
        pts = []
        for nid in sg.nodes():
            key = str(nid)
            if key in full_layout:
                pts.append(full_layout[key])
        if not pts:
            continue
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        cg_key = str(cid)
        if cg_key not in cg_pos:
            continue
        drl_centroids.append([cx, cy])
        target_centroids.append(list(cg_pos[cg_key]))

    if len(drl_centroids) < 2:
        return full_layout

    A = np.array(drl_centroids)
    B = np.array(target_centroids)

    # Center both
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A0 = A - a_mean
    B0 = B - b_mean

    # Optimal rotation (Kabsch / orthogonal Procrustes)
    H = A0.T @ B0
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, np.sign(d)])  # correct reflection
    R = Vt.T @ S @ U.T

    # Optimal uniform scale
    scale = np.trace(B0.T @ (A0 @ R.T)) / np.trace(A0.T @ A0)

    # Apply: transformed = (pos - a_mean) @ R.T * scale + b_mean
    all_ids = list(full_layout.keys())
    coords = np.array([full_layout[k] for k in all_ids])
    transformed = (coords - a_mean) @ R.T * scale + b_mean

    # Normalize to [0, 1]
    mins = transformed.min(axis=0)
    maxs = transformed.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    transformed = (transformed - mins) / ranges

    return {
        nid: [round(float(transformed[i, 0]), 6), round(float(transformed[i, 1]), 6)]
        for i, nid in enumerate(all_ids)
    }


# ---------------------------------------------------------------------------
# JS data export
# ---------------------------------------------------------------------------

def export_clusters_js(
    output_dir: Path, cg: nx.Graph, names: dict[int, str],
    pos: dict, colors: dict,
) -> None:
    """Write clusters.js with cluster overview data."""
    nodes = []
    for node_id in cg.nodes():
        attrs = cg.nodes[node_id]
        nid = int(node_id) if isinstance(node_id, str) else node_id
        p = pos.get(node_id, (0.5, 0.5))
        show_count = int(attrs.get("size", 0))
        nodes.append({
            "id": nid,
            "label": attrs.get("label", names.get(nid, f"Cluster {nid}")),
            "x": round(p[0], 4),
            "y": round(p[1], 4),
            "showCount": show_count,
            "size": max(5, show_count / 40),
            "difficulty": round(float(attrs.get("difficulty", 0.5)), 4),
            "topShows": attrs.get("top_shows", ""),
            "color": colors.get(node_id, "#888888"),
        })

    edges = []
    for u, v, attrs in cg.edges(data=True):
        w = float(attrs.get("weight", 0))
        max_w = max(float(a.get("weight", 0)) for _, _, a in cg.edges(data=True)) or 1
        edges.append({
            "source": int(u) if isinstance(u, str) else u,
            "target": int(v) if isinstance(v, str) else v,
            "weight": round(w, 4),
            "edgeCount": int(attrs.get("edge_count", 0)),
            "size": round(0.5 + 3.0 * (w / max_w), 2),
        })

    data = {"nodes": nodes, "edges": edges}
    path = output_dir / "data" / "clusters.js"
    with open(path, "w") as f:
        f.write("var CLUSTERS_DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print(f"  Wrote {path}", file=sys.stderr)


def export_cluster_js(
    output_dir: Path, cid: int, sg: nx.Graph, name: str,
    pos: dict, colors: dict,
) -> tuple[int, int]:
    """Write cluster_N.js with per-cluster show data. Returns (node_count, edge_count)."""
    nodes = []

    # Compute token range for size normalization
    all_tokens = [int(sg.nodes[n].get("total_tokens", 0)) for n in sg.nodes()]
    max_tok = max(all_tokens) if all_tokens else 1

    for node_id in sg.nodes():
        attrs = sg.nodes[node_id]
        p = pos.get(node_id, (0.5, 0.5))
        tokens = int(attrs.get("total_tokens", 0))
        nodes.append({
            "id": str(node_id),
            "label": attrs.get("title", attrs.get("label", str(node_id))),
            "x": round(p[0], 4),
            "y": round(p[1], 4),
            "category": attrs.get("category", ""),
            "episodes": int(attrs.get("episode_count", 0)),
            "centrality": round(float(attrs.get("centrality_score", 0)), 4),
            "globalCentrality": round(float(attrs.get("global_centrality", 0)), 4),
            "tokens": tokens,
            "color": colors.get(node_id, "#888888"),
            "size": round(2 + 10 * (tokens / max_tok), 2),
        })

    edges = []
    for u, v, attrs in sg.edges(data=True):
        edges.append({
            "source": str(u),
            "target": str(v),
            "weight": round(float(attrs.get("weight", 0)), 4),
        })

    data = {
        "clusterName": name,
        "clusterId": cid,
        "nodes": nodes,
        "edges": edges,
    }
    path = output_dir / "data" / f"cluster_{cid}.js"
    with open(path, "w") as f:
        f.write("var CLUSTER_DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print(f"  Wrote {path} ({len(nodes)} nodes, {len(edges)} edges)", file=sys.stderr)
    return len(nodes), len(edges)


def export_search_index_js(output_dir: Path, subgraphs: dict[int, nx.Graph]) -> None:
    """Write search_index.js with all shows for text search."""
    entries = []
    for cid, sg in sorted(subgraphs.items()):
        for node_id in sg.nodes():
            attrs = sg.nodes[node_id]
            entries.append({
                "id": str(node_id),
                "title": attrs.get("title", attrs.get("label", str(node_id))),
                "cluster": cid,
                "category": attrs.get("category", ""),
            })

    path = output_dir / "data" / "search_index.js"
    with open(path, "w") as f:
        f.write("var SEARCH_INDEX = ")
        json.dump(entries, f, separators=(",", ":"))
        f.write(";\n")
    print(f"  Wrote {path} ({len(entries)} entries)", file=sys.stderr)


def export_pathfinding_js(output_dir: Path, full_graph: dict) -> None:
    """Write pathfinding.js with full graph adjacency."""
    path = output_dir / "data" / "pathfinding.js"
    with open(path, "w") as f:
        f.write("var PATHFINDING_DATA = ")
        json.dump(full_graph, f, separators=(",", ":"))
        f.write(";\n")
    print(f"  Wrote {path}", file=sys.stderr)


def export_fullgraph_js(
    output_dir: Path,
    subgraphs: dict[int, nx.Graph],
    names: dict[int, str],
    full_graph: dict,
    full_layout: dict[str, list[float]],
) -> tuple[int, int]:
    """Write fullgraph.js using the precomputed full graph layout from graph.py.

    Uses the real DrL layout positions and a single global distance-from-center
    RdYlGn gradient — same style as the per-cluster pages but across the
    entire graph.

    Returns (total_nodes, total_edges).
    """
    cluster_ids = sorted(subgraphs.keys())

    # Build node_id -> cluster_id mapping
    node_to_cluster: dict[str, int] = {}
    for cid, sg in subgraphs.items():
        for node_id in sg.nodes():
            node_to_cluster[str(node_id)] = cid

    # Collect all positions that have a layout entry
    positioned = {nid: pos for nid, pos in full_layout.items()
                  if nid in node_to_cluster}

    # Global distance-from-center coloring (same as per-cluster pages)
    colors = color_by_distance_from_center(positioned)

    # Global token max for sizing
    all_tokens = []
    for sg in subgraphs.values():
        for n in sg.nodes():
            all_tokens.append(int(sg.nodes[n].get("total_tokens", 0)))
    max_tok = max(all_tokens) if all_tokens else 1

    nodes = []
    for nid, (x, y) in positioned.items():
        cid = node_to_cluster[nid]
        sg = subgraphs[cid]
        attrs = sg.nodes[nid] if sg.has_node(nid) else {}
        tokens = int(attrs.get("total_tokens", 0))
        nodes.append({
            "id": nid,
            "label": attrs.get("title", attrs.get("label", nid)),
            "x": round(x, 4),
            "y": round(y, 4),
            "category": attrs.get("category", ""),
            "cluster": cid,
            "clusterName": names.get(cid, f"Cluster {cid}"),
            "episodes": int(attrs.get("episode_count", 0)),
            "tokens": tokens,
            "color": colors.get(nid, "#888888"),
            "size": round(1.5 + 5 * (tokens / max_tok), 2),
        })

    # Build edges from full_graph adjacency
    adj = full_graph.get("adj", {})
    node_id_set = {n["id"] for n in nodes}
    edges = []
    seen = set()
    for src_id, neighbors in adj.items():
        if src_id not in node_id_set:
            continue
        for target_id, weight in neighbors:
            tgt = str(target_id)
            if tgt not in node_id_set:
                continue
            edge_key = (min(src_id, tgt), max(src_id, tgt))
            if edge_key in seen:
                continue
            seen.add(edge_key)
            edges.append({
                "source": src_id,
                "target": tgt,
                "weight": round(float(weight), 4),
            })

    data = {
        "nodes": nodes,
        "edges": edges,
    }
    path = output_dir / "data" / "fullgraph.js"
    with open(path, "w") as f:
        f.write("var FULLGRAPH_DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print(
        f"  Wrote {path} ({len(nodes)} nodes, {len(edges)} edges)",
        file=sys.stderr,
    )
    return len(nodes), len(edges)


def export_composite_js(
    output_dir: Path,
    subgraphs: dict[int, nx.Graph],
    names: dict[int, str],
    full_graph: dict,
    cg_pos: dict,
    cluster_layouts: dict[int, dict],
    cluster_colors: dict[int, dict],
) -> tuple[int, int]:
    """Write composite.js with composite layout of all nodes.

    Uses cluster-level positions as centers, offsets each node by its
    within-cluster layout position (scaled down). Per-cluster distance-from-
    center coloring + cluster hue legend.

    Returns (total_nodes, total_edges).
    """
    n_clusters = len(subgraphs)
    cluster_ids = sorted(subgraphs.keys())
    cluster_hue_colors = {}
    for i, cid in enumerate(cluster_ids):
        hue = i / max(n_clusters, 1)
        rgba = cm.hsv(hue)
        cluster_hue_colors[cid] = rgba_to_hex(rgba[0], rgba[1], rgba[2])

    CLUSTER_SPREAD = 10.0
    CLUSTER_RADIUS = 1.2

    nodes = []
    for cid in cluster_ids:
        sg = subgraphs[cid]
        cg_key = str(cid)
        cx, cy = cg_pos.get(cg_key, (0.5, 0.5))
        cx *= CLUSTER_SPREAD
        cy *= CLUSTER_SPREAD
        layout = cluster_layouts.get(cid, {})

        all_tokens = [int(sg.nodes[n].get("total_tokens", 0)) for n in sg.nodes()]
        max_tok = max(all_tokens) if all_tokens else 1

        node_colors = cluster_colors.get(cid, {})
        for node_id in sg.nodes():
            attrs = sg.nodes[node_id]
            lx, ly = layout.get(node_id, (0.5, 0.5))
            x = cx + (lx - 0.5) * CLUSTER_RADIUS
            y = cy + (ly - 0.5) * CLUSTER_RADIUS
            tokens = int(attrs.get("total_tokens", 0))
            nodes.append({
                "id": str(node_id),
                "label": attrs.get("title", attrs.get("label", str(node_id))),
                "x": round(x, 4),
                "y": round(y, 4),
                "category": attrs.get("category", ""),
                "cluster": cid,
                "clusterName": names.get(cid, f"Cluster {cid}"),
                "episodes": int(attrs.get("episode_count", 0)),
                "tokens": tokens,
                "color": node_colors.get(node_id, "#888888"),
                "size": round(1.5 + 5 * (tokens / max_tok), 2),
            })

    adj = full_graph.get("adj", {})
    node_id_set = {n["id"] for n in nodes}
    edges = []
    seen = set()
    for src_id, neighbors in adj.items():
        if src_id not in node_id_set:
            continue
        for target_id, weight in neighbors:
            tgt = str(target_id)
            if tgt not in node_id_set:
                continue
            edge_key = (min(src_id, tgt), max(src_id, tgt))
            if edge_key in seen:
                continue
            seen.add(edge_key)
            edges.append({
                "source": src_id,
                "target": tgt,
                "weight": round(float(weight), 4),
            })

    clusters_info = []
    for cid in cluster_ids:
        clusters_info.append({
            "id": cid,
            "name": names.get(cid, f"Cluster {cid}"),
            "color": cluster_hue_colors[cid],
        })

    data = {"nodes": nodes, "edges": edges, "clusters": clusters_info}
    path = output_dir / "data" / "composite.js"
    with open(path, "w") as f:
        f.write("var COMPOSITE_DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print(
        f"  Wrote {path} ({len(nodes)} nodes, {len(edges)} edges)",
        file=sys.stderr,
    )
    return len(nodes), len(edges)


# ---------------------------------------------------------------------------
# CSS generation
# ---------------------------------------------------------------------------

def write_css(output_dir: Path) -> None:
    """Write the main stylesheet."""
    css = """\
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { height: 100%; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: __BACKGROUND__; color: #333; }

#site-header {
  display: flex; align-items: center; justify-content: space-between;
  height: 48px; padding: 0 16px;
  background: #fff; border-bottom: 1px solid #ddd;
  z-index: 100; position: relative;
}
.header-left { display: flex; align-items: center; gap: 20px; }
.site-title { color: #c0392b; text-decoration: none; font-weight: 700; font-size: 16px; }
nav a { color: #666; text-decoration: none; font-size: 14px; margin-right: 12px; }
nav a:hover { color: #333; }
.header-right { position: relative; }

.search-container { position: relative; }
#global-search {
  width: 260px; padding: 6px 12px; border-radius: 4px;
  border: 1px solid #ccc; background: #fff; color: #333;
  font-size: 13px; outline: none;
}
#global-search:focus { border-color: #c0392b; }
.search-dropdown {
  display: none; position: absolute; top: 100%; right: 0;
  width: 360px; max-height: 400px; overflow-y: auto;
  background: #fff; border: 1px solid #ddd; border-radius: 4px;
  z-index: 200; margin-top: 4px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.search-item {
  display: block; padding: 8px 12px; text-decoration: none;
  border-bottom: 1px solid #eee;
}
.search-item:hover { background: #f0f0eb; }
.search-title { display: block; color: #333; font-size: 13px; }
.search-meta { display: block; color: #999; font-size: 11px; margin-top: 2px; }

#content { height: calc(100% - 48px); position: relative; }

/* Graph wrapper */
#graph-wrapper { width: 100%; height: 100%; position: relative; }
#sigma-container { width: 100%; height: 100%; }

/* Legend */
#legend {
  position: absolute; bottom: 20px; left: 20px;
  background: rgba(255, 255, 255, 0.92); padding: 10px 14px;
  border-radius: 6px; border: 1px solid #ddd;
  z-index: 50; min-width: 180px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.legend-title { font-size: 11px; color: #666; margin-bottom: 6px; }
.legend-bar {
  height: 12px; border-radius: 3px;
  background: linear-gradient(to right, #d73027, #fee08b, #1a9850);
}
.legend-labels {
  display: flex; justify-content: space-between;
  font-size: 10px; color: #999; margin-top: 3px;
}

/* Tooltip */
.tooltip {
  position: fixed; pointer-events: none;
  background: rgba(255, 255, 255, 0.96); color: #333;
  padding: 8px 12px; border-radius: 4px;
  border: 1px solid #ddd; font-size: 12px;
  max-width: 300px; z-index: 300; line-height: 1.5;
  box-shadow: 0 2px 8px rgba(0,0,0,0.12);
}

/* Cluster header */
#cluster-header {
  position: absolute; top: 0; left: 0; right: 0;
  padding: 8px 16px; background: rgba(255, 255, 255, 0.92);
  z-index: 50; display: flex; align-items: center; gap: 16px;
  border-bottom: 1px solid #ddd;
}
#cluster-header h1 { font-size: 16px; font-weight: 600; color: #333; }
.cluster-stats { color: #999; font-size: 13px; }
.back-btn {
  color: #c0392b; text-decoration: none; font-size: 13px;
  white-space: nowrap;
}
.back-btn:hover { text-decoration: underline; }

/* Detail panel */
.detail-panel {
  position: absolute; top: 0; right: 0; bottom: 0;
  width: 340px; background: #fff;
  border-left: 1px solid #ddd; overflow-y: auto;
  z-index: 60; padding: 16px;
}
.detail-close {
  position: absolute; top: 8px; right: 12px;
  background: none; border: none; color: #999; font-size: 22px; cursor: pointer;
}
.detail-close:hover { color: #333; }
.detail-panel h2 { font-size: 16px; margin-bottom: 10px; padding-right: 30px; color: #333; }
.detail-meta { font-size: 13px; line-height: 1.8; margin-bottom: 14px; color: #666; }
.detail-panel h3 { font-size: 13px; color: #999; margin-bottom: 8px; }
.neighbor-list { list-style: none; }
.neighbor-list li {
  display: flex; justify-content: space-between; align-items: center;
  padding: 4px 0; border-bottom: 1px solid #eee; font-size: 13px;
}
.neighbor-list a { color: #333; text-decoration: none; }
.neighbor-list a:hover { color: #c0392b; }
.nb-weight { color: #999; font-size: 11px; font-family: monospace; }

/* Pathfinder */
.pathfinder-layout {
  height: 100%; overflow-y: auto; padding: 30px;
  max-width: 800px; margin: 0 auto;
}
.pathfinder-form h1 { font-size: 22px; margin-bottom: 8px; color: #333; }
.pathfinder-info { color: #777; font-size: 13px; margin-bottom: 20px; line-height: 1.5; }
.pf-field { position: relative; margin-bottom: 16px; }
.pf-field label { display: block; font-size: 13px; color: #666; margin-bottom: 4px; }
.pf-field input[type="text"] {
  width: 100%; padding: 8px 12px; border-radius: 4px;
  border: 1px solid #ccc; background: #fff; color: #333;
  font-size: 14px; outline: none;
}
.pf-field input[type="text"]:focus { border-color: #c0392b; }
.pf-dropdown {
  display: none; position: absolute; top: 100%; left: 0; right: 0;
  max-height: 200px; overflow-y: auto;
  background: #fff; border: 1px solid #ddd; border-radius: 4px;
  z-index: 100; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.pf-dropdown-item {
  padding: 8px 12px; cursor: pointer; font-size: 13px;
  border-bottom: 1px solid #eee;
}
.pf-dropdown-item:hover { background: #f0f0eb; }
.pf-tag {
  display: inline-block; margin-top: 4px; padding: 3px 8px;
  background: #e8e8e3; color: #333; border-radius: 3px; font-size: 12px;
}
.pf-button {
  padding: 10px 24px; background: #c0392b; color: #fff;
  border: none; border-radius: 4px; font-size: 14px;
  cursor: pointer; font-weight: 600;
}
.pf-button:hover { background: #a93226; }
.pf-button:disabled { background: #ccc; cursor: not-allowed; color: #999; }

.pathfinder-results { margin-top: 24px; }
.pathfinder-results h2 { font-size: 18px; margin-bottom: 14px; color: #333; }
.path-card {
  background: #fff; border: 1px solid #ddd; border-radius: 6px;
  padding: 14px; margin-bottom: 12px;
}
.path-header { font-size: 13px; color: #999; margin-bottom: 8px; }
.path-steps { line-height: 2; }
.path-step a { color: #333; text-decoration: none; }
.path-step a:hover { color: #c0392b; text-decoration: underline; }
.cluster-badge {
  display: inline-block; padding: 1px 5px; margin-left: 3px;
  background: #e8e8e3; border-radius: 3px; font-size: 10px; color: #999;
}
.path-arrow { color: #bbb; }
.no-paths { color: #999; font-style: italic; }

/* Cluster color legend (full graph page) */
.cluster-legend { max-height: 50vh; overflow-y: auto; }
.cluster-legend .legend-title { margin-bottom: 8px; }
.legend-item { display: flex; align-items: center; gap: 6px; padding: 2px 0; font-size: 11px; color: #555; }
.legend-swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }
.legend-label { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* Documentation pages */
.docs-layout {
  display: flex; height: 100%; overflow: hidden;
}
.docs-sidebar {
  width: 220px; flex-shrink: 0; background: #fff; border-right: 1px solid #ddd;
  padding: 20px 0; overflow-y: auto;
}
.docs-nav-title {
  font-size: 13px; font-weight: 700; color: #333; padding: 0 16px; margin-bottom: 12px;
}
.docs-nav-link {
  display: block; padding: 6px 16px; color: #666; text-decoration: none;
  font-size: 13px; border-left: 3px solid transparent;
}
.docs-nav-link:hover { color: #333; background: #f5f5f2; }
.docs-nav-link.active {
  color: #c0392b; border-left-color: #c0392b; font-weight: 600; background: #fdf6f5;
}
.docs-content {
  flex: 1; overflow-y: auto; padding: 32px 40px; max-width: 820px;
  line-height: 1.7; font-size: 14px; color: #333;
}
.docs-content h1 {
  font-size: 24px; font-weight: 700; margin-bottom: 16px; color: #222;
}
.docs-content h2 {
  font-size: 18px; font-weight: 600; margin-top: 32px; margin-bottom: 12px;
  padding-bottom: 6px; border-bottom: 1px solid #eee; color: #222;
}
.docs-content h3 {
  font-size: 15px; font-weight: 600; margin-top: 20px; margin-bottom: 8px; color: #333;
}
.docs-content p { margin-bottom: 12px; }
.docs-content ul, .docs-content ol {
  margin-bottom: 12px; padding-left: 24px;
}
.docs-content li { margin-bottom: 4px; }
.docs-content code {
  background: #f0f0eb; padding: 1px 5px; border-radius: 3px;
  font-size: 12px; color: #c0392b;
}
.docs-content strong { font-weight: 600; }
.docs-content a { color: #c0392b; text-decoration: none; }
.docs-content a:hover { text-decoration: underline; }
.docs-content dl { margin-bottom: 16px; }
.docs-content dt {
  font-weight: 600; margin-top: 12px; color: #222;
}
.docs-content dd { margin-left: 0; margin-bottom: 8px; color: #555; }

/* Docs table of contents */
.docs-toc {
  background: #f9f9f7; border: 1px solid #eee; border-radius: 6px;
  padding: 12px 16px; margin-bottom: 24px;
}
.docs-toc-title { font-size: 12px; font-weight: 600; color: #999; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
.docs-toc a {
  display: block; color: #555; text-decoration: none; font-size: 13px;
  padding: 2px 0;
}
.docs-toc a:hover { color: #c0392b; }

/* Docs card grid */
.docs-card-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 20px 0;
}
.docs-card {
  background: #fff; border: 1px solid #ddd; border-radius: 6px;
  padding: 16px; text-decoration: none; color: #333;
  transition: border-color 0.15s, box-shadow 0.15s;
}
.docs-card:hover {
  border-color: #c0392b; box-shadow: 0 2px 8px rgba(192,57,43,0.08);
  text-decoration: none;
}
.docs-card h3 { font-size: 15px; font-weight: 600; margin-bottom: 6px; color: #c0392b; }
.docs-card p { font-size: 13px; color: #666; margin: 0; line-height: 1.5; }

/* Docs definitions */
.docs-definitions dt { font-size: 15px; }
.docs-definitions dd { padding-left: 16px; border-left: 2px solid #eee; }

/* Docs metric cards */
.docs-metric-card {
  background: #f9f9f7; border: 1px solid #eee; border-radius: 6px;
  padding: 12px 16px; margin-bottom: 16px; display: flex; align-items: center; gap: 16px;
}
.docs-metric-example {
  font-family: monospace; font-size: 18px; font-weight: 700; color: #c0392b;
  white-space: nowrap;
}
.docs-metric-card p { margin: 0; font-size: 13px; color: #666; }

/* Docs table */
.docs-table {
  width: 100%; border-collapse: collapse; margin-bottom: 16px; font-size: 13px;
}
.docs-table th {
  text-align: left; padding: 8px 12px; background: #f5f5f2;
  border-bottom: 2px solid #ddd; font-weight: 600; color: #333;
}
.docs-table td { padding: 8px 12px; border-bottom: 1px solid #eee; }
.docs-table code { font-size: 12px; }

/* Docs formula */
.docs-formula {
  background: #f9f9f7; border: 1px solid #eee; border-radius: 6px;
  padding: 14px 18px; margin: 16px 0; font-size: 13px;
}
.formula-row { padding: 4px 0; display: flex; gap: 12px; }
.formula-label { color: #666; min-width: 220px; }
.formula-math { font-family: monospace; color: #333; font-weight: 500; }

/* Docs color helpers */
.docs-color-green { color: #1a9850; }
.docs-color-yellow { color: #b8860b; }
.docs-color-red { color: #d73027; }

.docs-color-bar { margin: 12px 0 16px; max-width: 300px; }

/* Docs schema diagram */
.docs-schema-diagram {
  display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  margin: 20px 0; justify-content: center;
}
.schema-table {
  background: #fff; border: 2px solid #ddd; border-radius: 6px;
  min-width: 180px; overflow: hidden;
}
.schema-table-name {
  background: #c0392b; color: #fff; font-weight: 600; font-size: 13px;
  padding: 6px 12px; text-align: center;
}
.schema-columns { padding: 6px 0; }
.schema-col { padding: 3px 12px; font-size: 12px; font-family: monospace; color: #333; }
.schema-pk {
  display: inline-block; background: #f0c040; color: #333; font-size: 9px;
  padding: 0 4px; border-radius: 2px; margin-right: 4px; font-family: sans-serif; font-weight: 600;
}
.schema-fk {
  display: inline-block; background: #5dade2; color: #fff; font-size: 9px;
  padding: 0 4px; border-radius: 2px; margin-right: 4px; font-family: sans-serif; font-weight: 600;
}
.schema-arrow { color: #999; font-size: 13px; }

.cluster-badge-inline {
  display: inline-block; padding: 1px 5px;
  background: #e8e8e3; border-radius: 3px; font-size: 10px; color: #999;
}
"""
    css = css.replace("__BACKGROUND__", THEME["background"])
    path = output_dir / "assets" / "style.css"
    with open(path, "w") as f:
        f.write(css)
    print(f"  Wrote {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# JS library download
# ---------------------------------------------------------------------------

def download_js_libs(output_dir: Path) -> None:
    """Download Sigma.js and Graphology UMD bundles."""
    assets = output_dir / "assets"
    for filename, url in JS_LIBS.items():
        dest = assets / filename
        if dest.exists():
            print(f"  {filename} already exists, skipping", file=sys.stderr)
            continue
        print(f"  Downloading {filename}...", file=sys.stderr)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SiteCompiler/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            with open(dest, "wb") as f:
                f.write(data)
            print(f"  Saved {dest} ({len(data):,} bytes)", file=sys.stderr)
        except Exception as e:
            print(
                f"  WARNING: Could not download {filename}: {e}\n"
                f"  Please download manually from {url}\n"
                f"  and place in {assets}/",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def render_templates(
    output_dir: Path, names: dict[int, str],
    cluster_stats: dict[int, tuple[int, int]],
    fullgraph_stats: tuple[int, int] | None = None,
    composite_stats: tuple[int, int] | None = None,
) -> None:
    """Render Jinja2 templates to HTML files."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=False,
    )

    # Index page
    tpl = env.get_template("index.html")
    html = tpl.render(root="", theme=THEME)
    with open(output_dir / "index.html", "w") as f:
        f.write(html)
    print(f"  Wrote {output_dir / 'index.html'}", file=sys.stderr)

    # Pathfinder page
    tpl = env.get_template("pathfinder.html")
    html = tpl.render(root="", theme=THEME)
    with open(output_dir / "pathfinder.html", "w") as f:
        f.write(html)
    print(f"  Wrote {output_dir / 'pathfinder.html'}", file=sys.stderr)

    # Full graph page
    if fullgraph_stats:
        tpl = env.get_template("fullgraph.html")
        fg_nc, fg_ec = fullgraph_stats
        html = tpl.render(
            root="",
            theme=THEME,
            node_count=fg_nc,
            edge_count=fg_ec,
            cluster_count=len(cluster_stats),
        )
        with open(output_dir / "fullgraph.html", "w") as f:
            f.write(html)
        print(f"  Wrote {output_dir / 'fullgraph.html'}", file=sys.stderr)

    # Composite view page
    if composite_stats:
        tpl = env.get_template("composite.html")
        cp_nc, cp_ec = composite_stats
        html = tpl.render(
            root="",
            theme=THEME,
            node_count=cp_nc,
            edge_count=cp_ec,
            cluster_count=len(cluster_stats),
        )
        with open(output_dir / "composite.html", "w") as f:
            f.write(html)
        print(f"  Wrote {output_dir / 'composite.html'}", file=sys.stderr)

    # Per-cluster pages
    tpl = env.get_template("cluster.html")
    for cid in sorted(cluster_stats.keys()):
        nc, ec = cluster_stats[cid]
        html = tpl.render(
            root="../",
            cluster_id=cid,
            cluster_name=names.get(cid, f"Cluster {cid}"),
            node_count=nc,
            edge_count=ec,
            theme=THEME,
        )
        with open(output_dir / "cluster" / f"{cid}.html", "w") as f:
            f.write(html)
    print(
        f"  Wrote {len(cluster_stats)} cluster pages to {output_dir / 'cluster'}/",
        file=sys.stderr,
    )

    # Documentation pages
    docs_pages = [
        ("docs_overview.html", "docs/index.html", "overview"),
        ("docs_methodology.html", "docs/methodology.html", "methodology"),
        ("docs_metrics.html", "docs/metrics.html", "metrics"),
        ("docs_views.html", "docs/views.html", "views"),
        ("docs_database.html", "docs/database.html", "database"),
    ]
    for template_name, out_path, active_doc in docs_pages:
        tpl = env.get_template(template_name)
        html = tpl.render(root="../", theme=THEME, active_doc=active_doc)
        with open(output_dir / out_path, "w") as f:
            f.write(html)
    print(
        f"  Wrote {len(docs_pages)} docs pages to {output_dir / 'docs'}/",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile graph data into an interactive HTML site."
    )
    parser.add_argument(
        "-i", "--input", type=Path, required=True,
        help="Input directory with GraphML files, names.json, full_graph.json",
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True,
        help="Output directory for the generated site (e.g. www/)",
    )
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    # Create output directories
    for d in ["", "cluster", "data", "assets", "docs"]:
        (output_dir / d).mkdir(parents=True, exist_ok=True)

    print("Loading graph data...", file=sys.stderr)
    cg = load_cluster_graph(input_dir)
    subgraphs = load_cluster_subgraphs(input_dir)
    names = load_names(input_dir)
    full_graph = load_full_graph(input_dir)
    full_layout = load_full_layout(input_dir)

    print(
        f"Loaded cluster graph ({cg.number_of_nodes()} clusters) "
        f"and {len(subgraphs)} subgraphs",
        file=sys.stderr,
    )

    # Compute cluster-level layout & colors
    print("Computing cluster layout...", file=sys.stderr)
    cg_pos = compute_layout(cg, weight="weight", seed=42, k=1.5, iterations=200)
    cg_colors = color_by_distance_from_center(cg_pos)

    # Export cluster overview data
    print("Exporting data files...", file=sys.stderr)
    export_clusters_js(output_dir, cg, names, cg_pos, cg_colors)

    # Compute per-cluster layouts & export
    cluster_stats = {}
    cluster_layouts = {}
    cluster_colors = {}
    for cid, sg in sorted(subgraphs.items()):
        n = sg.number_of_nodes()
        print(f"  Cluster {cid}: {n} nodes...", file=sys.stderr)
        k = 2.0 / max(1, n ** 0.4)
        pos = compute_layout(sg, weight="weight", seed=42, k=k, iterations=100)
        colors = color_by_distance_from_center(pos)
        nc, ec = export_cluster_js(output_dir, cid, sg, names.get(cid, ""), pos, colors)
        cluster_stats[cid] = (nc, ec)
        cluster_layouts[cid] = pos
        cluster_colors[cid] = colors

    # Search index
    export_search_index_js(output_dir, subgraphs)

    # Pathfinding data
    fullgraph_stats = None
    composite_stats = None
    if full_graph:
        export_pathfinding_js(output_dir, full_graph)
        # Composite view (cluster-level positions + per-cluster layouts)
        print("Exporting composite visualization...", file=sys.stderr)
        composite_stats = export_composite_js(
            output_dir, subgraphs, names, full_graph,
            cg_pos, cluster_layouts, cluster_colors,
        )
        # Full graph (real DrL layout, aligned to cluster overview)
        if full_layout:
            print("Aligning full graph layout to cluster positions...", file=sys.stderr)
            aligned_layout = align_layout_to_clusters(full_layout, subgraphs, cg_pos)
            print("Exporting full graph visualization...", file=sys.stderr)
            fullgraph_stats = export_fullgraph_js(
                output_dir, subgraphs, names, full_graph, aligned_layout,
            )
        else:
            print(
                "  WARNING: full_layout.json not found. Full graph page will not be generated.\n"
                "  Re-run graph.py to generate it.",
                file=sys.stderr,
            )
    else:
        print(
            "  WARNING: full_graph.json not found. Pathfinding will not work.\n"
            "  Re-run graph.py to generate it.",
            file=sys.stderr,
        )
        # Write empty placeholder
        path = output_dir / "data" / "pathfinding.js"
        with open(path, "w") as f:
            f.write('var PATHFINDING_DATA = {"nodes":{},"adj":{}};\n')

    # Download JS libraries
    print("Setting up assets...", file=sys.stderr)
    download_js_libs(output_dir)
    write_css(output_dir)

    # Render HTML
    print("Rendering HTML pages...", file=sys.stderr)
    render_templates(output_dir, names, cluster_stats, fullgraph_stats, composite_stats)

    total_shows = sum(nc for nc, _ in cluster_stats.values())
    print(
        f"\nDone! Site generated at {output_dir}/\n"
        f"  {len(cluster_stats)} cluster pages, {total_shows} shows\n"
        f"  Open {output_dir}/index.html in a browser (file:// works)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
