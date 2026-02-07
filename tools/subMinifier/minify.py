#!/usr/bin/env python3
"""
Subtitle Minifier — strips formatting/tags from .srt, .ass, and .ssa subtitle
files and outputs clean text with one sentence per line.

Usage:
    # Minify a single file (prints to stdout)
    python minify.py path/to/file.srt

    # Minify a single file to an output file
    python minify.py path/to/file.srt -o output.txt

    # Minify an entire directory tree (recreates structure under output dir)
    python minify.py path/to/subs/ -o path/to/clean_subs/

    # Only process .ass files
    python minify.py path/to/subs/ -o out/ --ext .ass

    # Dry-run: show what would be processed
    python minify.py path/to/subs/ -o out/ --dry-run
"""

import argparse
import os
import re
import sys
from pathlib import Path

SUPPORTED_EXTENSIONS = {".srt", ".ass", ".ssa"}

# ASS/SSA override tags like {\pos(960,540)}, {\fs38.2}, {\b1}, {\an8}, {\fad(200,300)}, etc.
RE_ASS_TAGS = re.compile(r"\{[^}]*\}")

# SRT supports limited HTML-like tags: <b>, <i>, <u>, <font color="...">, etc.
RE_HTML_TAGS = re.compile(r"<[^>]+>")

# Timestamps for SRT: "00:01:23,456 --> 00:01:25,789"
RE_SRT_TIMESTAMP = re.compile(
    r"^\d{1,2}:\d{2}:\d{2}[,.]\d{2,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{2,3}"
)

# Purely numeric lines (subtitle index in SRT)
RE_NUMERIC_LINE = re.compile(r"^\d+\s*$")


def detect_encoding(raw: bytes) -> str:
    """Detect encoding from BOM or fall back to utf-8."""
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if raw.startswith(b"\xfe\xff"):
        return "utf-16-be"
    return "utf-8"


def read_file(path: Path) -> str:
    raw = path.read_bytes()
    encoding = detect_encoding(raw)
    try:
        return raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return raw.decode("utf-8", errors="replace")


def clean_ass_line(text: str) -> str:
    """Strip ASS/SSA override tags and convert line-break markers to newlines."""
    text = RE_ASS_TAGS.sub("", text)
    # \N and \n are ASS soft/hard line breaks
    text = text.replace("\\N", "\n").replace("\\n", "\n")
    # \h is a hard space
    text = text.replace("\\h", " ")
    return text


def minify_ass(content: str) -> list[str]:
    """Extract dialogue text from ASS/SSA content."""
    lines: list[str] = []
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        # Dialogue and Comment lines carry subtitle text.
        # Format: "Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text"
        if stripped.startswith("Dialogue:") or stripped.startswith("Comment:"):
            # Text field is everything after the 9th comma
            parts = stripped.split(",", 9)
            if len(parts) < 10:
                continue
            text = parts[9]
            text = clean_ass_line(text)
            for sub_line in text.split("\n"):
                sub_line = sub_line.strip()
                if sub_line:
                    lines.append(sub_line)
    return lines


def minify_srt(content: str) -> list[str]:
    """Extract dialogue text from SRT content."""
    lines: list[str] = []
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        # Skip numeric index lines and timestamp lines
        if RE_NUMERIC_LINE.match(stripped):
            continue
        if RE_SRT_TIMESTAMP.match(stripped):
            continue
        # Strip any HTML-style tags and ASS-style tags (some SRT files include them)
        cleaned = RE_HTML_TAGS.sub("", stripped)
        cleaned = RE_ASS_TAGS.sub("", cleaned)
        cleaned = cleaned.strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def minify_file(path: Path) -> list[str]:
    """Minify a single subtitle file and return clean lines."""
    content = read_file(path)
    ext = path.suffix.lower()
    if ext in (".ass", ".ssa"):
        return minify_ass(content)
    if ext == ".srt":
        return minify_srt(content)
    return []


def process_single(src: Path, dst: Path | None) -> None:
    """Process one file: write to dst or stdout."""
    lines = minify_file(src)
    output = "\n".join(lines) + "\n" if lines else ""
    if dst:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)


def gather_files(root: Path, extensions: set[str]) -> list[Path]:
    """Recursively collect subtitle files under root."""
    files: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            p = Path(dirpath) / fname
            if p.suffix.lower() in extensions:
                files.append(p)
    files.sort()
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip formatting from subtitle files, producing clean one-sentence-per-line text."
    )
    parser.add_argument("input", type=Path, help="Input subtitle file or directory")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output file (for single input) or directory (for directory input). "
             "Omit to print to stdout (single file only).",
    )
    parser.add_argument(
        "--ext", nargs="*", default=None,
        help="Only process these extensions (e.g. --ext .srt .ass). Default: all supported.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List files that would be processed without writing anything.",
    )
    args = parser.parse_args()

    extensions = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.ext} if args.ext else SUPPORTED_EXTENSIONS

    src: Path = args.input.resolve()

    if src.is_file():
        if args.dry_run:
            print(src)
            return
        process_single(src, args.output)
        return

    if not src.is_dir():
        print(f"Error: {src} is not a file or directory", file=sys.stderr)
        sys.exit(1)

    if not args.output:
        print("Error: --output is required when input is a directory", file=sys.stderr)
        sys.exit(1)

    files = gather_files(src, extensions)
    if not files:
        print("No subtitle files found.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        for f in files:
            print(f)
        print(f"\n{len(files)} files would be processed.")
        return

    out_root = args.output.resolve()
    processed = 0
    errors = 0
    for f in files:
        rel = f.relative_to(src)
        dst = out_root / rel.with_suffix(".txt")
        try:
            process_single(f, dst)
            processed += 1
        except Exception as e:
            print(f"Error processing {f}: {e}", file=sys.stderr)
            errors += 1

    print(f"Done. {processed} files minified, {errors} errors.", file=sys.stderr)


if __name__ == "__main__":
    main()
