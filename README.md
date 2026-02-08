# Japanese Show Graph

Build a similarity graph of Japanese TV shows and movies based on morphological analysis of their subtitle files. Raw subtitles from a [kitsunekko](https://kitsunekko.net/) mirror are cleaned, tokenized with SudachiPy into morpheme frequency tables in SQLite, then compared via TF-IDF cosine similarity, clustered into vocabulary domains, and exported as an interactive website.

## Prerequisites

- **Python 3.12** (SudachiPy has no prebuilt wheel for 3.14)
- **pip packages** (steps 1-3):
  ```
  sudachipy sudachidict_core
  scipy scikit-learn sparse-dot-topn igraph leidenalg networkx matplotlib
  ```
- **pip packages** (step 4 — site compiler):
  ```
  networkx jinja2 matplotlib scipy
  ```
- **Raw subtitle data**: the `kitsunekko-mirror-main.zip` archive, extracted to `subs/kitsunekko-mirror-main/`

Install all dependencies:

```sh
pip3.12 install sudachipy sudachidict_core scipy scikit-learn sparse-dot-topn igraph leidenalg networkx matplotlib jinja2
```

## Pipeline

### Step 1 — Minify subtitles

Strips formatting and tags from `.srt`, `.ass`, and `.ssa` files, producing clean one-sentence-per-line text.

```sh
python3.12 tools/subMinifier/minify.py subs/kitsunekko-mirror-main/subtitles/ -o subs/minSubs/
```

Options:
- `--ext .srt .ass` — only process specific extensions
- `--dry-run` — list files without writing

### Step 2 — Parse into database

Tokenizes the minified text with SudachiPy and populates a SQLite database with per-show morpheme frequency data.

```sh
python3.12 tools/subParser/parse.py subs/minSubs/ -o subs/japaneseShowGraph.db
```

This step takes a long time (~11,000 shows). You can resume an interrupted run:

```sh
python3.12 tools/subParser/parse.py subs/minSubs/ -o subs/japaneseShowGraph.db --resume
```

Options:
- `--mode A/B/C` — SudachiPy segmentation mode (default: B, balanced)
- `--resume` — skip shows already in the database
- `--limit N` — process only the first N shows (for testing)
- `--min-kana-ratio 0.1` — minimum kana token ratio to accept a show
- `--min-tokens 50` — minimum total tokens to accept a show

**Filtering (parser-side):**

The parser applies several layers of filtering before data reaches the database:

1. **Non-Japanese filename filtering** — files with language codes in their name (`.en.`, `[eng]`, `.zh.`, `[ko]`, etc.) are skipped automatically to avoid ingesting translated subtitle tracks.
2. **Line-level Japanese filtering** — only lines containing at least one Japanese character (hiragana, katakana, or kanji) are kept; pure-ASCII or other-language lines are dropped.
3. **POS filtering at tokenization** — supplementary symbols (`補助記号`) and whitespace (`空白`) are discarded during tokenization and never stored.
4. **OOV tracking** — morphemes not found in the SudachiPy dictionary are flagged with `is_oov = 1` in the database for downstream filtering.
5. **Show-level validation** — a show is rejected entirely if it has fewer than `--min-tokens` total tokens (default 50) or its kana token ratio falls below `--min-kana-ratio` (default 10%). This catches non-Japanese or nearly-empty subtitle sets that slipped through earlier checks.

### Step 3 — Build graph

Reads the SQLite database, computes TF-IDF vectors and cosine similarity, clusters shows with the Leiden algorithm, and exports GraphML files and JSON data.

```sh
python3.12 tools/grapher/graph.py subs/japaneseShowGraph.db -o output/
```

Options:
- `--topk 20` — top-k most similar neighbours per show (default: 20)
- `--threshold 0.01` — minimum cosine similarity to retain (default: 0.01)
- `--resolution 1.0` — Leiden resolution; higher = more clusters (default: 1.0)
- `--threads 4` — threads for similarity computation (default: 1)
- `--min-tokens 500` — minimum total tokens to include a show (default: 500)
- `--min-kana-ratio 0.1` — minimum kana token ratio to include a show (default: 0.1)
- `--plot` — also generate PNG visualizations
- `--names output/names.json` — JSON file mapping cluster IDs to human-readable names

**Filtering (grapher-side):**

The grapher applies its own filtering on top of the parser's, since it needs stricter thresholds for meaningful similarity:

1. **Morpheme POS filtering** — the following part-of-speech categories are excluded from TF-IDF vectors:
   - `名詞-固有名詞` (proper nouns) — character/place names that are too show-specific to reflect general vocabulary
   - `名詞-数詞` (numerals) — numbers carry no language-learning signal
   - `記号` (symbols/punctuation) — non-semantic tokens
   - `感動詞` (interjections) — low-frequency emotional particles
2. **OOV exclusion** — morphemes flagged `is_oov` by the parser (not in the SudachiPy dictionary) are excluded, removing garbled text and non-standard tokens.
3. **Show-level quality filtering** — shows with fewer than `--min-tokens` total tokens (default 500, stricter than the parser's 50) or a kana ratio below `--min-kana-ratio` (default 10%) are dropped from the graph entirely.

Output: `cluster_graph.graphml`, per-cluster `cluster_N.graphml` files, `full_graph.json`, and `full_layout.json`.

### Step 4 — Compile website

Generates a static HTML site with interactive Sigma.js graph visualizations, search, and pathfinding. Works via `file://` protocol (no server needed).

```sh
python3.12 siteCompiler/compile.py -i output/ -o www/
```

Then open `www/index.html` in a browser.

## Project Structure

```
japaneseShowGraph/
├── tools/
│   ├── subMinifier/minify.py     # Step 1: strip formatting from subtitles
│   ├── subParser/parse.py        # Step 2: SudachiPy tokenization -> SQLite
│   └── grapher/graph.py          # Step 3: TF-IDF -> similarity -> clusters -> GraphML
├── siteCompiler/
│   ├── compile.py                # Step 4: GraphML -> interactive HTML site
│   └── templates/                # Jinja2 HTML templates
├── subs/                         # (gitignored) subtitle data and database
├── output/                       # (gitignored) GraphML and JSON graph files
└── www/                          # (gitignored) generated website
```
