#!/usr/bin/env python3
"""
Grapher — reads the morpheme frequency database and produces a two-level
graph structure exported as GraphML files for Gephi.

Level 1: Cluster graph (vocabulary domains as nodes, inter-domain edges)
Level 2: Per-cluster subgraphs (shows as nodes within each cluster)

Usage:
    python tools/grapher/graph.py subs/japaneseShowGraph.db -o output/
    python tools/grapher/graph.py subs/japaneseShowGraph.db -o output/ --topk 20 --threshold 0.01
    python tools/grapher/graph.py subs/japaneseShowGraph.db -o output/ --resolution 1.0
    python tools/grapher/graph.py subs/japaneseShowGraph.db -o output/ --threads 4

Dependencies:
    pip install scipy scikit-learn sparse-dot-topn igraph leidenalg networkx matplotlib
"""

import argparse
import json
import sqlite3
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import igraph as ig
import leidenalg
import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sparse_dot_topn import sp_matmul_topn

KANA_RE = re.compile(r'[\u3040-\u309f\u30a0-\u30ff]')


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

@dataclass
class ShowMeta:
    title: str
    category: str
    episode_count: int
    total_tokens: int = 0
    kana_tokens: int = 0


@dataclass
class IndexMaps:
    """Bidirectional maps between SQLite IDs and matrix indices."""
    show_db_to_idx: dict[int, int] = field(default_factory=dict)
    show_idx_to_db: dict[int, int] = field(default_factory=dict)
    morph_db_to_idx: dict[int, int] = field(default_factory=dict)
    morph_idx_to_db: dict[int, int] = field(default_factory=dict)


def load_data(db_path: str) -> tuple[csr_matrix, IndexMaps, dict[int, ShowMeta], dict[int, str]]:
    """Load the database and build a sparse count matrix + metadata.

    Returns (count_matrix, maps, shows_meta, morph_labels) where morph_labels
    maps morpheme DB ID → dictionary_form for cluster naming.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # Show metadata
    shows_meta: dict[int, ShowMeta] = {}
    for row in conn.execute("SELECT id, title, category, episode_count FROM shows"):
        shows_meta[row["id"]] = ShowMeta(
            title=row["title"],
            category=row["category"] or "",
            episode_count=row["episode_count"] or 0,
        )

    # Morpheme labels + POS for cluster naming and filtering
    morph_labels: dict[int, str] = {}
    proper_noun_ids: set[int] = set()
    kana_morph_ids: set[int] = set()
    for row in conn.execute(
        "SELECT id, surface_form, dictionary_form, part_of_speech FROM morphemes"
    ):
        morph_labels[row["id"]] = row["dictionary_form"]
        if row["part_of_speech"].startswith("名詞-固有名詞"):
            proper_noun_ids.add(row["id"])
        if KANA_RE.search(row["surface_form"]):
            kana_morph_ids.add(row["id"])

    print(
        f"Filtering {len(proper_noun_ids):,} proper noun morphemes "
        f"(character names, place names, etc.)",
        file=sys.stderr,
    )

    # Build ID maps and COO data in a single pass, skipping proper nouns
    maps = IndexMaps()
    rows, cols, data = [], [], []

    cur = conn.execute("SELECT show_id, morpheme_id, count FROM show_morphemes")
    for show_id, morpheme_id, count in cur:
        # Always count total tokens (including proper nouns)
        shows_meta[show_id].total_tokens += count
        if morpheme_id in kana_morph_ids:
            shows_meta[show_id].kana_tokens += count

        # Skip proper nouns for the similarity matrix
        if morpheme_id in proper_noun_ids:
            continue

        if show_id not in maps.show_db_to_idx:
            idx = len(maps.show_db_to_idx)
            maps.show_db_to_idx[show_id] = idx
            maps.show_idx_to_db[idx] = show_id
        if morpheme_id not in maps.morph_db_to_idx:
            idx = len(maps.morph_db_to_idx)
            maps.morph_db_to_idx[morpheme_id] = idx
            maps.morph_idx_to_db[idx] = morpheme_id

        row_idx = maps.show_db_to_idx[show_id]
        col_idx = maps.morph_db_to_idx[morpheme_id]
        rows.append(row_idx)
        cols.append(col_idx)
        data.append(count)

    conn.close()

    n_shows = len(maps.show_db_to_idx)
    n_morphs = len(maps.morph_db_to_idx)
    count_matrix = coo_matrix(
        (data, (rows, cols)), shape=(n_shows, n_morphs)
    ).tocsr()

    print(
        f"Loaded {n_shows} shows × {n_morphs} morphemes "
        f"({count_matrix.nnz:,} nonzeros)",
        file=sys.stderr,
    )
    return count_matrix, maps, shows_meta, morph_labels


# ---------------------------------------------------------------------------
# 1b. Filter low-quality shows
# ---------------------------------------------------------------------------

def filter_shows(
    count_matrix: csr_matrix,
    maps: IndexMaps,
    shows_meta: dict[int, ShowMeta],
    min_tokens: int,
    min_kana_ratio: float,
) -> tuple[csr_matrix, IndexMaps]:
    """Remove shows below quality thresholds (too few tokens or non-Japanese)."""
    keep_indices = []
    n_low_tokens = 0
    n_low_kana = 0

    for idx in range(count_matrix.shape[0]):
        db_id = maps.show_idx_to_db[idx]
        meta = shows_meta[db_id]
        total = max(meta.total_tokens, 1)
        kana_ratio = meta.kana_tokens / total

        if meta.total_tokens < min_tokens:
            n_low_tokens += 1
            continue
        if kana_ratio < min_kana_ratio:
            n_low_kana += 1
            continue
        keep_indices.append(idx)

    n_original = count_matrix.shape[0]
    n_removed = n_original - len(keep_indices)
    if n_removed:
        print(
            f"Filtered {n_removed} shows: "
            f"{n_low_tokens} below {min_tokens} tokens, "
            f"{n_low_kana} below {min_kana_ratio:.0%} kana ratio",
            file=sys.stderr,
        )

    filtered_matrix = count_matrix[keep_indices]
    new_maps = IndexMaps()
    for new_idx, old_idx in enumerate(keep_indices):
        db_id = maps.show_idx_to_db[old_idx]
        new_maps.show_db_to_idx[db_id] = new_idx
        new_maps.show_idx_to_db[new_idx] = db_id
    new_maps.morph_db_to_idx = maps.morph_db_to_idx
    new_maps.morph_idx_to_db = maps.morph_idx_to_db

    print(
        f"Kept {len(keep_indices)}/{n_original} shows after filtering",
        file=sys.stderr,
    )
    return filtered_matrix, new_maps


# ---------------------------------------------------------------------------
# 2. TF-IDF
# ---------------------------------------------------------------------------

def compute_tfidf(count_matrix: csr_matrix) -> csr_matrix:
    """Transform raw counts into L2-normed TF-IDF vectors."""
    transformer = TfidfTransformer(
        norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True,
    )
    tfidf = transformer.fit_transform(count_matrix)
    print(f"TF-IDF matrix: {tfidf.shape}, nnz={tfidf.nnz:,}", file=sys.stderr)
    return tfidf


# ---------------------------------------------------------------------------
# 3. Top-k similarity
# ---------------------------------------------------------------------------

def compute_topk_similarity(
    tfidf: csr_matrix, top_k: int, threshold: float, n_threads: int,
) -> csr_matrix:
    """Compute sparse cosine similarity keeping top-k per row above threshold."""
    sim = sp_matmul_topn(
        tfidf, tfidf.T, top_n=top_k, threshold=threshold, n_threads=n_threads,
    )
    # Symmetrize, zero diagonal, clean up
    sim = sim.maximum(sim.T)
    sim.setdiag(0)
    sim.eliminate_zeros()
    print(
        f"Similarity matrix: {sim.shape}, nnz={sim.nnz:,} "
        f"(density {sim.nnz / sim.shape[0]**2:.6f})",
        file=sys.stderr,
    )
    return sim


# ---------------------------------------------------------------------------
# 4. Build igraph
# ---------------------------------------------------------------------------

def build_igraph(similarity: csr_matrix) -> ig.Graph:
    """Build an undirected weighted graph from the upper triangle of similarity."""
    sim_upper = similarity.copy()
    # Keep only upper triangle
    rows, cols = sim_upper.nonzero()
    mask = rows < cols
    edges = list(zip(rows[mask].tolist(), cols[mask].tolist()))
    weights = np.asarray(sim_upper[rows[mask], cols[mask]]).flatten().tolist()

    g = ig.Graph(n=similarity.shape[0], edges=edges, directed=False)
    g.es["weight"] = weights
    print(
        f"Graph: {g.vcount()} vertices, {g.ecount()} edges",
        file=sys.stderr,
    )
    return g


# ---------------------------------------------------------------------------
# 4b. Compute full graph layout
# ---------------------------------------------------------------------------

def compute_full_layout(
    graph: ig.Graph,
    maps: IndexMaps,
) -> dict[str, list[float]]:
    """Compute a 2D layout for the full graph using igraph's DrL algorithm.

    DrL (Distributed Recursive Layout / OpenOrd) is designed for large graphs
    (thousands of nodes). Returns {db_id_str: [x, y]} normalized to [0, 1].
    """
    print("Computing full graph layout (DrL)...", file=sys.stderr)
    layout = graph.layout_drl(weights="weight")
    coords = np.array(layout.coords)

    # Normalize to [0, 1]
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    coords = (coords - mins) / ranges

    positions: dict[str, list[float]] = {}
    for idx in range(graph.vcount()):
        db_id = maps.show_idx_to_db[idx]
        positions[str(db_id)] = [round(float(coords[idx, 0]), 6),
                                  round(float(coords[idx, 1]), 6)]

    print(f"Layout computed for {len(positions)} nodes", file=sys.stderr)
    return positions


# ---------------------------------------------------------------------------
# 5. Community detection
# ---------------------------------------------------------------------------

def detect_communities(graph: ig.Graph, resolution: float) -> list[int]:
    """Leiden community detection with RBConfigurationVertexPartition."""
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=42,
    )
    membership = partition.membership
    n_clusters = len(set(membership))
    print(f"Detected {n_clusters} communities", file=sys.stderr)
    return membership


# ---------------------------------------------------------------------------
# 6. Centrality
# ---------------------------------------------------------------------------

def compute_centrality(
    similarity: csr_matrix, membership: list[int],
) -> list[float]:
    """Weighted degree centrality within each cluster.

    For each show, sums its similarity edge weights to other shows in the
    same cluster. Rank-normalized per-cluster to 0-1 so the full color
    range is used evenly (avoids outlier skew from min-max).
    """
    n_shows = similarity.shape[0]
    centrality = np.zeros(n_shows)
    clusters = defaultdict(list)
    for idx, cid in enumerate(membership):
        clusters[cid].append(idx)

    for cid, members in clusters.items():
        sub = similarity[np.ix_(members, members)]
        weighted_deg = np.asarray(sub.sum(axis=1)).flatten()

        # Rank-based normalization: evenly distributed 0-1
        n = len(members)
        ranks = weighted_deg.argsort().argsort()  # rank positions
        normalized = ranks / max(n - 1, 1)

        for i, idx in enumerate(members):
            centrality[idx] = float(normalized[i])

    return centrality.tolist()


def compute_global_centrality(similarity: csr_matrix) -> list[float]:
    """Weighted degree centrality across the full similarity graph.

    Each show's score is its total connection strength to all other shows.
    High weighted degree = shares vocabulary with many shows = central = easier.
    Low weighted degree = isolated/specialized vocabulary = peripheral = harder.
    """
    weighted_deg = np.asarray(similarity.sum(axis=1)).flatten()
    n = len(weighted_deg)
    ranks = weighted_deg.argsort().argsort()
    normalized = ranks / max(n - 1, 1)

    print(
        f"Global centrality: min={normalized.min():.3f}, "
        f"max={normalized.max():.3f}, mean={normalized.mean():.3f}",
        file=sys.stderr,
    )
    return normalized.tolist()


# ---------------------------------------------------------------------------
# 6b. Cluster names from top discriminative morphemes
# ---------------------------------------------------------------------------

def load_cluster_names(names_path: Path | None) -> dict[int, str]:
    """Load cluster names from a JSON file, or return empty dict."""
    if names_path and names_path.exists():
        with open(names_path) as f:
            raw = json.load(f)
        return {int(k): str(v) for k, v in raw.items()}
    return {}


def print_cluster_summary(
    membership: list[int],
    maps: IndexMaps,
    shows_meta: dict[int, ShowMeta],
    cluster_names: dict[int, str],
) -> None:
    """Print a summary of each cluster for manual review."""
    clusters: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(membership):
        clusters[cid].append(idx)

    for cid in sorted(clusters.keys()):
        members = clusters[cid]
        name = cluster_names.get(cid, "(unnamed)")

        # Top shows by token count
        member_info = []
        for idx in members:
            db_id = maps.show_idx_to_db[idx]
            m = shows_meta[db_id]
            member_info.append((m.total_tokens, m.title, m.category))
        member_info.sort(reverse=True)

        # Category breakdown
        cats: Counter[str] = Counter()
        for _, _, cat in member_info:
            cats[cat] += 1
        cat_str = ", ".join(f"{c}:{n}" for c, n in cats.most_common())

        top5 = [f"{t} ({c})" for _, t, c in member_info[:5]]
        print(
            f"  Cluster {cid} ({len(members)} shows): {name}\n"
            f"    Categories: {cat_str}\n"
            f"    Top: {', '.join(top5)}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# 7. Cluster graph edges
# ---------------------------------------------------------------------------

@dataclass
class ClusterEdge:
    weight: float  # mean similarity of cross-cluster edges
    edge_count: int  # number of cross-cluster edges


def build_cluster_graph(
    similarity: csr_matrix, membership: list[int],
) -> dict[tuple[int, int], ClusterEdge]:
    """Aggregate cross-cluster edges into cluster-level edges."""
    pair_sum: dict[tuple[int, int], float] = defaultdict(float)
    pair_count: dict[tuple[int, int], int] = defaultdict(int)

    rows, cols = similarity.nonzero()
    vals = np.asarray(similarity[rows, cols]).flatten()

    for r, c, v in zip(rows.tolist(), cols.tolist(), vals.tolist()):
        cr, cc = membership[r], membership[c]
        if cr == cc:
            continue
        key = (min(cr, cc), max(cr, cc))
        pair_sum[key] += v
        pair_count[key] += 1

    edges: dict[tuple[int, int], ClusterEdge] = {}
    for key in pair_sum:
        # Each edge counted twice (symmetric matrix), halve count
        count = pair_count[key] // 2
        total = pair_sum[key] / 2
        avg = total / count if count > 0 else 0.0
        edges[key] = ClusterEdge(weight=avg, edge_count=count)

    print(f"Cluster graph: {len(edges)} inter-cluster edges", file=sys.stderr)
    return edges


# ---------------------------------------------------------------------------
# 8. Export GraphML
# ---------------------------------------------------------------------------

def export_graphml(
    output_dir: Path,
    maps: IndexMaps,
    shows_meta: dict[int, ShowMeta],
    similarity: csr_matrix,
    membership: list[int],
    centrality: list[float],
    global_centrality: list[float],
    cluster_edges: dict[tuple[int, int], ClusterEdge],
    cluster_names: dict[int, str],
) -> None:
    """Write cluster graph and per-cluster subgraphs as GraphML files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group shows by cluster
    clusters: dict[int, list[int]] = defaultdict(list)  # cid -> [matrix idx]
    for idx, cid in enumerate(membership):
        clusters[cid].append(idx)

    # --- Cluster graph ---
    cg = nx.Graph()
    for cid, members in sorted(clusters.items()):
        # Top 3 shows by total_tokens for reference
        member_meta = []
        for idx in members:
            db_id = maps.show_idx_to_db[idx]
            member_meta.append((shows_meta[db_id].total_tokens, shows_meta[db_id].title))
        member_meta.sort(reverse=True)
        top_titles = " | ".join(t for _, t in member_meta[:3])

        avg_global = sum(global_centrality[i] for i in members) / len(members)
        cg.add_node(
            cid,
            label=cluster_names.get(cid, f"Cluster {cid}"),
            top_shows=top_titles,
            size=len(members),
            difficulty=round(avg_global, 6),
        )

    for (ca, cb), ce in cluster_edges.items():
        cg.add_edge(
            ca, cb,
            weight=ce.weight,
            distance=1.0 - ce.weight,
            edge_count=ce.edge_count,
        )

    cluster_path = output_dir / "cluster_graph.graphml"
    nx.write_graphml(cg, str(cluster_path))
    print(f"Wrote {cluster_path}", file=sys.stderr)

    # --- Per-cluster subgraphs ---
    for cid, members in sorted(clusters.items()):
        sg = nx.Graph()

        # Add show nodes
        for idx in members:
            db_id = maps.show_idx_to_db[idx]
            meta = shows_meta[db_id]
            sg.add_node(
                db_id,
                label=meta.title,
                title=meta.title,
                category=meta.category,
                episode_count=meta.episode_count,
                centrality_score=round(centrality[idx], 6),
                global_centrality=round(global_centrality[idx], 6),
                total_tokens=meta.total_tokens,
            )

        # Add intra-cluster edges from similarity matrix
        member_set = set(members)
        for idx in members:
            row = similarity.getrow(idx)
            _, col_indices = row.nonzero()
            for col_idx in col_indices.tolist():
                if col_idx in member_set and idx < col_idx:
                    w = float(similarity[idx, col_idx])
                    db_a = maps.show_idx_to_db[idx]
                    db_b = maps.show_idx_to_db[col_idx]
                    sg.add_edge(
                        db_a, db_b,
                        weight=w,
                        distance=round(1.0 - w, 6),
                    )

        sub_path = output_dir / f"cluster_{cid}.graphml"
        nx.write_graphml(sg, str(sub_path))
        print(
            f"Wrote {sub_path} ({sg.number_of_nodes()} shows, "
            f"{sg.number_of_edges()} edges)",
            file=sys.stderr,
        )

    # --- Full graph adjacency for pathfinding (all edges, intra + cross) ---
    nodes_out: dict[str, dict] = {}
    for idx in range(similarity.shape[0]):
        db_id = maps.show_idx_to_db[idx]
        meta = shows_meta[db_id]
        nodes_out[str(db_id)] = {
            "t": meta.title, "c": membership[idx], "cat": meta.category,
        }

    adj_out: dict[str, list] = {}
    for idx in range(similarity.shape[0]):
        db_id = maps.show_idx_to_db[idx]
        row = similarity.getrow(idx)
        _, col_indices = row.nonzero()
        neighbors = []
        for col_idx in col_indices.tolist():
            if col_idx != idx:
                nb_db = maps.show_idx_to_db[col_idx]
                w = round(float(similarity[idx, col_idx]), 4)
                neighbors.append([nb_db, w])
        if neighbors:
            neighbors.sort(key=lambda x: x[1], reverse=True)
            adj_out[str(db_id)] = neighbors

    full_path = output_dir / "full_graph.json"
    with open(full_path, "w") as f:
        json.dump({"nodes": nodes_out, "adj": adj_out}, f, separators=(",", ":"))
    print(f"Wrote {full_path} ({len(nodes_out)} nodes, full adjacency)", file=sys.stderr)


def export_full_layout(output_dir: Path, full_layout: dict[str, list[float]]) -> None:
    """Write full_layout.json with precomputed positions for every node."""
    path = output_dir / "full_layout.json"
    with open(path, "w") as f:
        json.dump(full_layout, f, separators=(",", ":"))
    print(f"Wrote {path} ({len(full_layout)} positions)", file=sys.stderr)


# ---------------------------------------------------------------------------
# 9. Plot
# ---------------------------------------------------------------------------

def plot_graphs(
    output_dir: Path,
    maps: IndexMaps,
    shows_meta: dict[int, ShowMeta],
    similarity: csr_matrix,
    membership: list[int],
    centrality: list[float],
    global_centrality: list[float],
    cluster_edges: dict[tuple[int, int], ClusterEdge],
    cluster_names: dict[int, str],
) -> None:
    """Render cluster graph and per-cluster subgraphs as PNG images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    matplotlib.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group shows by cluster
    clusters: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(membership):
        clusters[cid].append(idx)

    # --- Cluster graph ---
    cg = nx.Graph()
    for cid, members in sorted(clusters.items()):
        # Difficulty = average global centrality of member shows
        # Higher = more typical/shared vocabulary = easier entry
        avg_global = sum(global_centrality[i] for i in members) / len(members)
        cg.add_node(cid, size=len(members),
                    name=cluster_names.get(cid, f"Cluster {cid}"),
                    difficulty=avg_global)

    for (ca, cb), ce in cluster_edges.items():
        cg.add_edge(ca, cb, weight=ce.weight)

    fig, ax = plt.subplots(figsize=(22, 16))

    # Spring layout: well-connected clusters naturally gravitate to center
    pos = nx.spring_layout(cg, weight="weight", seed=42, k=1.5, iterations=200)

    # Color by distance from layout center (rank-normalized)
    center = np.mean(list(pos.values()), axis=0)
    nodes_list = list(cg.nodes())
    dists = np.array([np.linalg.norm(pos[n] - center) for n in nodes_list])
    n_nodes = len(dists)
    dist_ranks = dists.argsort().argsort()
    # Invert: close to center = 1.0 (green/easy), far = 0.0 (red/hard)
    norm_diffs = [1.0 - dist_ranks[i] / max(n_nodes - 1, 1) for i in range(n_nodes)]

    sizes = [cg.nodes[n]["size"] for n in nodes_list]
    max_size = max(sizes) if sizes else 1
    node_sizes = [600 + 4000 * (s / max_size) for s in sizes]

    # Draw ALL edges with width proportional to weight
    all_weights = [cg.edges[e]["weight"] for e in cg.edges()]
    max_ew = max(all_weights) if all_weights else 1
    all_widths = [0.5 + 3.5 * (w / max_ew) for w in all_weights]
    all_alphas = [0.15 + 0.45 * (w / max_ew) for w in all_weights]
    for edge, width, alpha in zip(cg.edges(), all_widths, all_alphas):
        nx.draw_networkx_edges(cg, pos, edgelist=[edge], ax=ax,
                               width=width, alpha=alpha, edge_color="#555")

    colormap = cm.RdYlGn
    node_colors = [colormap(nd) for nd in norm_diffs]
    nx.draw_networkx_nodes(cg, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, alpha=0.85, edgecolors="#333", linewidths=0.5)
    labels = {}
    for n in nodes_list:
        name = cg.nodes[n]["name"]
        labels[n] = f"{name}\n({cg.nodes[n]['size']} shows)"
    nx.draw_networkx_labels(cg, pos, labels, ax=ax, font_size=7)

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("Distance from center (green = easier)", fontsize=10)

    ax.set_title(f"Cluster Graph — {len(clusters)} domains, {sum(sizes)} shows", fontsize=15)
    ax.axis("off")
    fig.tight_layout()
    path = output_dir / "cluster_graph.png"
    fig.savefig(str(path), dpi=200)
    plt.close(fig)
    print(f"Wrote {path}", file=sys.stderr)

    # --- Per-cluster subgraphs ---
    colormap = cm.RdYlGn  # red=low centrality, green=high
    for cid, members in sorted(clusters.items()):
        sg = nx.Graph()
        for idx in members:
            db_id = maps.show_idx_to_db[idx]
            meta = shows_meta[db_id]
            sg.add_node(db_id, title=meta.title, centrality=centrality[idx],
                        total_tokens=meta.total_tokens)

        member_set = set(members)
        for idx in members:
            row = similarity.getrow(idx)
            _, col_indices = row.nonzero()
            for col_idx in col_indices.tolist():
                if col_idx in member_set and idx < col_idx:
                    w = float(similarity[idx, col_idx])
                    sg.add_edge(maps.show_idx_to_db[idx], maps.show_idx_to_db[col_idx], weight=w)

        if sg.number_of_nodes() == 0:
            continue

        fig, ax = plt.subplots(figsize=(16, 12))
        # Spring layout: well-connected shows naturally gravitate to center
        pos = nx.spring_layout(sg, weight="weight", seed=42,
                               k=2.0 / max(1, len(members)**0.4), iterations=100)

        # Color by distance from layout center (rank-normalized)
        center = np.mean(list(pos.values()), axis=0)
        nodes_list = list(sg.nodes())
        sg_dists = np.array([np.linalg.norm(pos[n] - center) for n in nodes_list])
        n_sg = len(sg_dists)
        sg_ranks = sg_dists.argsort().argsort()
        cents = [1.0 - sg_ranks[i] / max(n_sg - 1, 1) for i in range(n_sg)]

        tokens = [sg.nodes[n]["total_tokens"] for n in nodes_list]
        max_tok = max(tokens) if tokens else 1
        node_sizes = [40 + 460 * (t / max_tok) for t in tokens]
        node_colors = [colormap(c) for c in cents]

        nx.draw_networkx_edges(sg, pos, ax=ax, alpha=0.1, edge_color="#aaa", width=0.5)
        nx.draw_networkx_nodes(sg, pos, ax=ax, node_size=node_sizes, node_color=node_colors, alpha=0.85)

        # Label top 10 closest to center
        ranked = sorted(range(n_sg), key=lambda i: sg_dists[i])
        top_labels = {nodes_list[i]: sg.nodes[nodes_list[i]]["title"][:20] for i in ranked[:10]}
        nx.draw_networkx_labels(sg, pos, top_labels, ax=ax, font_size=6)

        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Centrality (ease of entry)", fontsize=9)

        cname = cluster_names.get(cid, "")
        ax.set_title(
            f"Cluster {cid}: {cname}\n{sg.number_of_nodes()} shows, {sg.number_of_edges()} edges",
            fontsize=12,
        )
        ax.axis("off")
        fig.tight_layout()
        path = output_dir / f"cluster_{cid}.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        print(f"Wrote {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build vocabulary domain graphs from the morpheme frequency database.",
    )
    parser.add_argument("db", type=str, help="Path to japaneseShowGraph.db")
    parser.add_argument(
        "-o", "--output", type=Path, required=True,
        help="Output directory for GraphML files",
    )
    parser.add_argument(
        "--topk", type=int, default=20,
        help="Top-k neighbours to keep per show in similarity (default: 20)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.01,
        help="Minimum cosine similarity to keep (default: 0.01)",
    )
    parser.add_argument(
        "--resolution", type=float, default=1.0,
        help="Leiden resolution parameter — higher = more clusters (default: 1.0)",
    )
    parser.add_argument(
        "--threads", type=int, default=1,
        help="Threads for similarity computation (default: 1)",
    )
    parser.add_argument(
        "--min-tokens", type=int, default=500,
        help="Minimum total tokens to include a show (default: 500)",
    )
    parser.add_argument(
        "--min-kana-ratio", type=float, default=0.1,
        help="Minimum kana token ratio to include a show (default: 0.1)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate PNG visualizations alongside GraphML files",
    )
    parser.add_argument(
        "--names", type=Path, default=None,
        help="JSON file mapping cluster IDs to names (e.g. output/names.json)",
    )
    args = parser.parse_args()

    # 1. Load
    count_matrix, maps, shows_meta, morph_labels = load_data(args.db)

    # 1b. Filter
    count_matrix, maps = filter_shows(
        count_matrix, maps, shows_meta, args.min_tokens, args.min_kana_ratio,
    )

    # 2. TF-IDF
    tfidf = compute_tfidf(count_matrix)
    del count_matrix

    # 3. Top-k similarity
    similarity = compute_topk_similarity(tfidf, args.topk, args.threshold, args.threads)

    # 3b. Global difficulty from graph structure
    global_centrality = compute_global_centrality(similarity)

    # 4. Build graph
    graph = build_igraph(similarity)

    # 4b. Full graph layout
    full_layout = compute_full_layout(graph, maps)

    # 5. Communities
    membership = detect_communities(graph, args.resolution)
    del graph

    # 6. Centrality (weighted degree within each cluster)
    centrality = compute_centrality(similarity, membership)
    del tfidf

    # 6b. Cluster names
    cluster_names = load_cluster_names(args.names)
    print_cluster_summary(membership, maps, shows_meta, cluster_names)

    # 7. Cluster edges
    cluster_edges = build_cluster_graph(similarity, membership)

    # 8. Export
    export_graphml(
        args.output, maps, shows_meta,
        similarity, membership, centrality, global_centrality,
        cluster_edges, cluster_names,
    )
    export_full_layout(args.output, full_layout)

    # 9. Plot
    if args.plot:
        plot_graphs(
            args.output, maps, shows_meta,
            similarity, membership, centrality, global_centrality,
            cluster_edges, cluster_names,
        )

    # Summary
    n_clusters = len(set(membership))
    print(f"\nDone. {n_clusters} clusters exported to {args.output}/", file=sys.stderr)


if __name__ == "__main__":
    main()
