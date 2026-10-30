#!/usr/bin/env python3
"""
Frequency list exporter — reads the morpheme frequency database and writes a
corpus-wide word frequency list as TSV.

Applies the same morpheme filtering as the grapher (drops OOV tokens, proper
nouns, numerals, symbols, interjections) and merges conjugated forms by grouping
on (dictionary_form, lexical POS). Shows are weighted equally by raw counts.

Usage:
    python3.12 tools/freqList/export_freq.py subs/japaneseShowGraph.db -o data/frequency_top20k.tsv
    python3.12 tools/freqList/export_freq.py subs/japaneseShowGraph.db -o data/frequency_top20k.tsv --top 20000
"""

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

EXCLUDED_POS_PREFIXES = ("名詞-固有名詞", "名詞-数詞", "記号", "感動詞")


def lexical_pos(pos: str) -> str:
    """First four POS fields (pos1-pos4), trailing wildcards stripped."""
    fields = pos.split("-")[:4]
    while fields and fields[-1] == "*":
        fields.pop()
    return "-".join(fields)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a word frequency list from the morpheme DB.")
    ap.add_argument("db", type=Path, help="path to japaneseShowGraph.db")
    ap.add_argument("-o", "--output", type=Path, required=True, help="output TSV path")
    ap.add_argument("--top", type=int, default=20000, help="number of entries to keep (default: 20000)")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    n_shows = conn.execute("SELECT COUNT(*) FROM shows").fetchone()[0]

    print("Aggregating per-morpheme totals...", file=sys.stderr)
    rows = conn.execute(
        """
        SELECT m.dictionary_form, m.part_of_speech, m.reading,
               SUM(sm.count) AS total, COUNT(sm.show_id) AS n_shows
        FROM show_morphemes sm
        JOIN morphemes m ON m.id = sm.morpheme_id
        WHERE m.is_oov = 0
        GROUP BY sm.morpheme_id
        """
    )

    # key: (dictionary_form, lexical_pos) -> totals, plus most frequent reading
    totals: dict[tuple[str, str], int] = defaultdict(int)
    show_counts: dict[tuple[str, str], int] = defaultdict(int)
    best_reading: dict[tuple[str, str], tuple[int, str]] = {}

    for dict_form, pos, reading, total, shows in rows:
        if pos.startswith(EXCLUDED_POS_PREFIXES):
            continue
        key = (dict_form, lexical_pos(pos))
        totals[key] += total
        # Show counts summed over surface variants may slightly exceed the true
        # distinct-show count; clamp to the corpus size.
        show_counts[key] = min(show_counts[key] + shows, n_shows)
        if reading and (key not in best_reading or total > best_reading[key][0]):
            best_reading[key] = (total, reading)

    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[: args.top]
    grand_total = sum(totals.values())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("rank\tlemma\treading\tpos\tcount\tshows\tper_million\n")
        for rank, ((dict_form, pos), total) in enumerate(ranked, start=1):
            reading = best_reading.get((dict_form, pos), ("", ""))[1]
            per_million = total / grand_total * 1_000_000
            f.write(f"{rank}\t{dict_form}\t{reading}\t{pos}\t{total}\t{show_counts[(dict_form, pos)]}\t{per_million:.2f}\n")

    size_kb = args.output.stat().st_size / 1024
    print(
        f"Wrote {args.output} ({len(ranked)} entries, {size_kb:.0f} KB) "
        f"from {n_shows} shows, {grand_total:,} filtered tokens",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
