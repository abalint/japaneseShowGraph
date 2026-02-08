#!/usr/bin/env python3
"""Diagnostic: understand why children's anime scores low on vocab commonality."""

import sqlite3
import sys
import numpy as np
from collections import defaultdict
from scipy.sparse import coo_matrix

# Reuse loading logic from graph.py
sys.path.insert(0, ".")
from graph import load_data, filter_shows, compute_vocab_difficulty

DB = sys.argv[1] if len(sys.argv) > 1 else "../../subs/japaneseShowGraph.db"

count_matrix, maps, shows_meta, morph_labels = load_data(DB)
count_matrix, maps = filter_shows(count_matrix, maps, shows_meta, 500, 0.1)

# Compute vocab difficulty
vocab_diff = compute_vocab_difficulty(count_matrix)

# Also compute the raw (pre-rank-normalization) scores for analysis
binary = count_matrix.copy()
binary.data = np.ones_like(binary.data)
df = np.asarray(binary.sum(axis=0)).flatten()
n_shows = count_matrix.shape[0]
norm_df = df / n_shows

raw_scores = np.zeros(n_shows)
for i in range(n_shows):
    row = count_matrix.getrow(i)
    _, col_indices = row.nonzero()
    counts = np.asarray(row[:, col_indices].todense()).flatten()
    dfs = norm_df[col_indices]
    total = counts.sum()
    if total > 0:
        raw_scores[i] = (counts * dfs).sum() / total

# Show per-category stats
cat_scores = defaultdict(list)
for idx in range(n_shows):
    db_id = maps.show_idx_to_db[idx]
    meta = shows_meta[db_id]
    cat_scores[meta.category].append((raw_scores[idx], vocab_diff[idx], meta.title))

print("\n=== Raw vocab commonality by category ===")
for cat in sorted(cat_scores.keys()):
    scores = cat_scores[cat]
    raw_vals = [s[0] for s in scores]
    rank_vals = [s[1] for s in scores]
    print(f"  {cat}: n={len(scores)}, raw_mean={np.mean(raw_vals):.4f}, "
          f"rank_mean={np.mean(rank_vals):.3f}")

# Now look at specific children's anime shows
# Find shows in cluster 12 (Kodomo) by looking at known titles
kodomo_titles = ["Crayon Shin-chan", "Pokemon", "Doraemon", "Alps no Shoujo Heidi",
                 "Sazae-san", "Chibi Maruko-chan"]

print("\n=== Sample children's shows ===")
for idx in range(n_shows):
    db_id = maps.show_idx_to_db[idx]
    meta = shows_meta[db_id]
    for kt in kodomo_titles:
        if kt.lower() in meta.title.lower():
            print(f"  {meta.title}: raw={raw_scores[idx]:.4f}, rank={vocab_diff[idx]:.3f}")

            # Show top morphemes by count and their DFs
            row = count_matrix.getrow(idx)
            _, col_indices = row.nonzero()
            counts = np.asarray(row[:, col_indices].todense()).flatten()
            dfs_vals = norm_df[col_indices]

            # Sort by count descending
            order = counts.argsort()[::-1]
            print(f"    Top 15 morphemes by count:")
            for j in order[:15]:
                morph_db_id = maps.morph_idx_to_db[col_indices[j]]
                label = morph_labels.get(morph_db_id, "?")
                print(f"      {label}: count={counts[j]:.0f}, df={dfs_vals[j]:.3f} "
                      f"({df[col_indices[j]]:.0f}/{n_shows} shows)")

            # Count of morphemes with low DF (< 0.05 = in <5% of shows)
            low_df_mask = dfs_vals < 0.05
            low_df_count = counts[low_df_mask].sum()
            total_count = counts.sum()
            print(f"    Tokens with df<5%: {low_df_count:.0f}/{total_count:.0f} "
                  f"({100*low_df_count/total_count:.1f}%)")
            break

# Compare with a "green" show from Modern Romance
print("\n=== Sample 'easy' shows (Modern Romance) ===")
romance_titles = ["Nigeru wa Haji", "Jizoku Kanou"]
for idx in range(n_shows):
    db_id = maps.show_idx_to_db[idx]
    meta = shows_meta[db_id]
    for rt in romance_titles:
        if rt.lower() in meta.title.lower():
            print(f"  {meta.title}: raw={raw_scores[idx]:.4f}, rank={vocab_diff[idx]:.3f}")
            row = count_matrix.getrow(idx)
            _, col_indices = row.nonzero()
            counts = np.asarray(row[:, col_indices].todense()).flatten()
            dfs_vals = norm_df[col_indices]
            low_df_mask = dfs_vals < 0.05
            low_df_count = counts[low_df_mask].sum()
            total_count = counts.sum()
            print(f"    Tokens with df<5%: {low_df_count:.0f}/{total_count:.0f} "
                  f"({100*low_df_count/total_count:.1f}%)")
            break

# Overall distribution
print(f"\n=== Raw score distribution ===")
print(f"  Min: {raw_scores.min():.4f}")
print(f"  25th: {np.percentile(raw_scores, 25):.4f}")
print(f"  Median: {np.median(raw_scores):.4f}")
print(f"  75th: {np.percentile(raw_scores, 75):.4f}")
print(f"  Max: {raw_scores.max():.4f}")
