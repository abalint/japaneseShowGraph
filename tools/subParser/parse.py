#!/usr/bin/env python3
"""
Subtitle Parser — tokenises minified subtitle text with SudachiPy and
populates a SQLite database with morpheme frequency data per show.

Usage:
    # Parse all shows
    python tools/subParser/parse.py subs/minSubs/ -o subs/japaneseShowGraph.db

    # Use SudachiPy mode C (longest segmentation)
    python tools/subParser/parse.py subs/minSubs/ -o subs/japaneseShowGraph.db --mode C

    # Resume an interrupted run
    python tools/subParser/parse.py subs/minSubs/ -o subs/japaneseShowGraph.db --resume

    # Process only the first 5 shows (for testing)
    python tools/subParser/parse.py subs/minSubs/ -o subs/japaneseShowGraph.db --limit 5

Dependencies:
    pip install sudachipy sudachidict_core
"""

import argparse
import os
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from sudachipy import Dictionary

# POS prefixes to skip: supplementary symbols (punctuation) and whitespace
SKIP_POS_PREFIXES = ("補助記号", "空白")

# ---------------------------------------------------------------------------
# Language filtering
# ---------------------------------------------------------------------------

# Hiragana, Katakana, CJK Unified Ideographs (covers kanji)
JAPANESE_RE = re.compile(
    r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]'
)
# Hiragana and Katakana only (kana is the definitive Japanese signal —
# Chinese shares kanji but never has kana)
KANA_RE = re.compile(r'[\u3040-\u309f\u30a0-\u30ff]')

# Filename patterns that indicate non-Japanese subtitle files.
# Matches language codes like .en., .eng., [eng], .zh., .chi., .ko., etc.
NON_JA_FILENAME_RE = re.compile(
    r'(?i)'
    r'(?:'
    r'\.(?:en|eng|english|zh|zho|chi|chinese|ko|kor|korean'
    r'|fr|fre|french|de|ger|german|es|spa|spanish'
    r'|pt|por|portuguese|it|ita|italian|ru|rus|russian'
    r'|ar|ara|arabic|th|tha|thai|vi|vie|vietnamese'
    r'|id|ind|indonesian|ms|msa|malay)\.'
    r'|'
    r'\[(?:en|eng|english|zh|zho|chi|chinese|ko|kor|korean'
    r'|fr|fre|french|de|ger|german|es|spa|spanish'
    r'|pt|por|portuguese|it|ita|italian|ru|rus|russian'
    r'|ar|ara|arabic|th|tha|thai|vi|vie|vietnamese'
    r'|id|ind|indonesian|ms|msa|malay)\]'
    r')'
)

# Default thresholds for show-level validation
DEFAULT_MIN_KANA_RATIO = 0.1   # at least 10% of morphemes should contain kana
DEFAULT_MIN_TOKENS = 50        # reject shows with very few tokens

SCHEMA = """\
CREATE TABLE IF NOT EXISTS morphemes (
    id INTEGER PRIMARY KEY,
    surface_form TEXT NOT NULL,
    dictionary_form TEXT NOT NULL,
    part_of_speech TEXT NOT NULL,
    reading TEXT,
    UNIQUE(surface_form, dictionary_form, part_of_speech, reading)
);

CREATE TABLE IF NOT EXISTS shows (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT,
    episode_count INTEGER,
    UNIQUE(title, category)
);

CREATE TABLE IF NOT EXISTS show_morphemes (
    show_id INTEGER NOT NULL,
    morpheme_id INTEGER NOT NULL,
    count INTEGER NOT NULL,
    PRIMARY KEY (show_id, morpheme_id),
    FOREIGN KEY (show_id) REFERENCES shows(id),
    FOREIGN KEY (morpheme_id) REFERENCES morphemes(id)
);

CREATE INDEX IF NOT EXISTS idx_show_morphemes_morpheme
    ON show_morphemes(morpheme_id);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    """Create the database and tables if they don't exist."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def discover_shows(root: Path) -> list[tuple[str, str, Path]]:
    """Walk minSubs/ and return (category, title, show_dir) for each show.

    Structure: root/category/show_name/  — where show_name is the immediate
    child of a category folder. Everything deeper is rolled into the same show.
    """
    shows: list[tuple[str, str, Path]] = []
    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for show_dir in sorted(category_dir.iterdir()):
            if not show_dir.is_dir():
                continue
            shows.append((category, show_dir.name, show_dir))
    return shows


def collect_txt_files(show_dir: Path) -> list[Path]:
    """Recursively collect all .txt files under a show directory,
    skipping files whose names indicate a non-Japanese language."""
    files: list[Path] = []
    skipped = 0
    for dirpath, _, filenames in os.walk(show_dir):
        for fname in sorted(filenames):
            p = Path(dirpath) / fname
            if p.suffix.lower() != ".txt":
                continue
            if NON_JA_FILENAME_RE.search(fname):
                skipped += 1
                continue
            files.append(p)
    if skipped:
        print(f"    Skipped {skipped} non-Japanese file(s) by filename", file=sys.stderr)
    return files


def is_japanese_line(line: str) -> bool:
    """Return True if the line contains at least one Japanese character
    (hiragana, katakana, or kanji/CJK ideograph)."""
    return bool(JAPANESE_RE.search(line))


def read_lines(files: list[Path]) -> list[str]:
    """Read and concatenate lines from all files, keeping only lines
    that contain Japanese script characters."""
    lines: list[str] = []
    total = 0
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                total += 1
                if is_japanese_line(stripped):
                    lines.append(stripped)
        except OSError as e:
            print(f"  Warning: could not read {f}: {e}", file=sys.stderr)
    filtered = total - len(lines)
    if filtered:
        print(f"    Filtered {filtered}/{total} non-Japanese lines", file=sys.stderr)
    return lines


def tokenise_lines(
    tokenizer, lines: list[str]
) -> Counter[tuple[str, str, str, str]]:
    """Tokenise lines and return morpheme counts.

    Returns a Counter keyed by (surface, dict_form, pos, reading).
    """
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for line in lines:
        for token in tokenizer.tokenize(line):
            pos_parts = token.part_of_speech()
            pos = "-".join(pos_parts)
            if pos_parts[0] in SKIP_POS_PREFIXES:
                continue
            surface = token.surface()
            dict_form = token.dictionary_form()
            reading = token.reading_form()
            counts[(surface, dict_form, pos, reading)] += 1
    return counts


def validate_japanese_content(
    morpheme_counts: Counter[tuple[str, str, str, str]],
    min_kana_ratio: float,
    min_tokens: int,
) -> tuple[bool, float, int]:
    """Check that morpheme data looks like Japanese content.

    Returns (is_valid, kana_ratio, total_tokens).
    A show passes if it has enough tokens and a sufficient proportion
    of morphemes whose surface form contains kana (hiragana/katakana).
    Kana is the definitive signal: Chinese shares kanji but never has kana.
    """
    total_tokens = sum(morpheme_counts.values())
    if total_tokens < min_tokens:
        return False, 0.0, total_tokens

    kana_tokens = sum(
        count for (surface, _, _, _), count in morpheme_counts.items()
        if KANA_RE.search(surface)
    )
    kana_ratio = kana_tokens / total_tokens if total_tokens else 0.0
    return kana_ratio >= min_kana_ratio, kana_ratio, total_tokens


def insert_show(
    conn: sqlite3.Connection,
    category: str,
    title: str,
    episode_count: int,
    morpheme_counts: Counter[tuple[str, str, str, str]],
) -> None:
    """Insert a show and its morpheme data in a single transaction."""
    cur = conn.cursor()
    cur.execute("BEGIN")

    cur.execute(
        "INSERT INTO shows (title, category, episode_count) VALUES (?, ?, ?)",
        (title, category, episode_count),
    )
    show_id = cur.lastrowid

    for (surface, dict_form, pos, reading), count in morpheme_counts.items():
        cur.execute(
            """INSERT INTO morphemes (surface_form, dictionary_form, part_of_speech, reading)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(surface_form, dictionary_form, part_of_speech, reading) DO NOTHING""",
            (surface, dict_form, pos, reading),
        )
        cur.execute(
            """SELECT id FROM morphemes
               WHERE surface_form = ? AND dictionary_form = ?
                 AND part_of_speech = ? AND reading = ?""",
            (surface, dict_form, pos, reading),
        )
        morpheme_id = cur.fetchone()[0]

        cur.execute(
            "INSERT INTO show_morphemes (show_id, morpheme_id, count) VALUES (?, ?, ?)",
            (show_id, morpheme_id, count),
        )

    cur.execute("COMMIT")


def existing_shows(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    """Return set of (title, category) already in the database."""
    cur = conn.execute("SELECT title, category FROM shows")
    return {(row[0], row[1]) for row in cur.fetchall()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse minified subtitle files with SudachiPy and populate a SQLite database."
    )
    parser.add_argument("input", type=Path, help="Root minified subtitles directory (e.g. subs/minSubs/)")
    parser.add_argument(
        "-o", "--output", type=Path, required=True,
        help="Output SQLite database path (e.g. subs/japaneseShowGraph.db)",
    )
    parser.add_argument(
        "--mode", choices=["A", "B", "C"], default="B",
        help="SudachiPy segmentation mode (default: B)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip shows already present in the database",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N shows (for testing)",
    )
    parser.add_argument(
        "--min-kana-ratio", type=float, default=DEFAULT_MIN_KANA_RATIO,
        help=f"Minimum kana token ratio to accept a show (default: {DEFAULT_MIN_KANA_RATIO})",
    )
    parser.add_argument(
        "--min-tokens", type=int, default=DEFAULT_MIN_TOKENS,
        help=f"Minimum total tokens to accept a show (default: {DEFAULT_MIN_TOKENS})",
    )
    args = parser.parse_args()

    root: Path = args.input.resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Initialise SudachiPy
    import sudachipy

    sudachi_dict = Dictionary()
    tokenizer = sudachi_dict.create()
    split_mode = {"A": sudachipy.SplitMode.A,
                  "B": sudachipy.SplitMode.B,
                  "C": sudachipy.SplitMode.C}[args.mode]

    class ModeTokenizer:
        """Wraps a SudachiPy tokenizer to always use a fixed SplitMode."""
        def __init__(self, tok, mode):
            self._tok = tok
            self._mode = mode
        def tokenize(self, text):
            return self._tok.tokenize(text, self._mode)

    tok = ModeTokenizer(tokenizer, split_mode)

    # Initialise database
    conn = init_db(args.output)

    # Discover shows
    shows = discover_shows(root)
    total = len(shows)
    if args.limit:
        shows = shows[:args.limit]
        total = len(shows)

    print(f"Found {total} shows to process.", file=sys.stderr)

    # If resuming, find which shows to skip
    skip = set()
    if args.resume:
        skip = existing_shows(conn)
        print(f"Resuming: {len(skip)} shows already in database.", file=sys.stderr)

    processed = 0
    skipped = 0
    rejected = 0
    errors = 0

    for i, (category, title, show_dir) in enumerate(shows, 1):
        if (title, category) in skip:
            skipped += 1
            continue

        try:
            txt_files = collect_txt_files(show_dir)
            if not txt_files:
                skipped += 1
                continue

            lines = read_lines(txt_files)
            if not lines:
                skipped += 1
                continue

            counts = tokenise_lines(tok, lines)

            # Validate that this actually looks like Japanese content
            is_ja, kana_ratio, total_tokens = validate_japanese_content(
                counts, args.min_kana_ratio, args.min_tokens,
            )
            if not is_ja:
                rejected += 1
                print(
                    f"[{i}/{total}] REJECTED {title} — "
                    f"kana ratio {kana_ratio:.1%}, {total_tokens} tokens",
                    file=sys.stderr,
                )
                continue

            insert_show(conn, category, title, len(txt_files), counts)
            processed += 1

            print(
                f"[{i}/{total}] {title} — {len(counts):,} morphemes, "
                f"kana {kana_ratio:.0%}",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"[{i}/{total}] Error processing {title}: {e}", file=sys.stderr)
            errors += 1

    conn.close()
    print(
        f"\nDone. {processed} processed, {skipped} skipped, "
        f"{rejected} rejected (non-Japanese), {errors} errors.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
