# Japanese Show Graph — Project State

## Overview

Build a graph/similarity database of Japanese TV shows and movies based on morphological analysis of their subtitle files. Raw subtitles are sourced from a kitsunekko mirror, cleaned into plain text, then parsed with SudachiPy into morpheme frequency tables in SQLite. TF-IDF vectors and show-to-show cosine similarity are computed from the database, then clustered into vocabulary domains and exported as GraphML graphs for Gephi.

## Pipeline

| Step | Tool | Status |
|------|------|--------|
| 0. Source data | `subs/kitsunekko-mirror-main/` (13 GB raw .srt/.ass/.ssa) | Done |
| 1. Minify subtitles | `tools/subMinifier/minify.py` → `subs/minSubs/` (5.3 GB clean .txt) | Done |
| 2. Parse & insert into DB | `tools/subParser/parse.py` → `subs/japaneseShowGraph.db` | **In progress** |
| 3. TF-IDF + similarity + graph | `tools/grapher/graph.py` → `output/*.graphml` + JSON | Done |
| 4. Compile website | `siteCompiler/compile.py` → `www/` (interactive Sigma.js site) | Available |

## Directory Structure

```
japaneseShowGraph/
├── kitsunekko-mirror-main.zip          # original archive
├── subs/
│   ├── kitsunekko-mirror-main/         # raw subtitle files (13 GB)
│   │   └── subtitles/{anime_movie,anime_tv,drama_movie,drama_tv,unsorted}/
│   ├── minSubs/                        # minified plain text (5.3 GB)
│   │   ├── anime_movie/   (712 shows)
│   │   ├── anime_tv/      (2,952 shows)
│   │   ├── drama_movie/   (3,565 shows)
│   │   ├── drama_tv/      (3,358 shows)
│   │   └── unsorted/      (417 shows)
│   └── japaneseShowGraph.db            # SQLite database (in progress)
├── tools/
│   ├── subMinifier/minify.py           # step 1: strip formatting from .srt/.ass/.ssa
│   ├── subParser/parse.py              # step 2: SudachiPy tokenization → SQLite
│   └── grapher/graph.py                # step 3: TF-IDF → similarity → clusters → GraphML
├── siteCompiler/
│   ├── compile.py                        # step 4: GraphML → interactive HTML site
│   └── templates/                        # Jinja2 HTML templates
├── output/                               # GraphML graph files + JSON data
│   ├── cluster_graph.graphml             # domain-level graph (clusters as nodes)
│   ├── cluster_N.graphml                 # per-cluster show graphs (N = 0..16)
│   ├── full_graph.json                   # complete adjacency with all edges + metadata
│   ├── full_layout.json                  # precomputed DrL layout positions
│   └── names.json                        # cluster ID → human-readable name
├── www/                                  # (gitignored) generated website
└── STATE.md
```

**Total shows across all categories: ~11,004**

## Database State

The parser is partway through `anime_movie`. Current counts:

| Table | Rows |
|-------|------|
| `shows` | 328 |
| `morphemes` | 141,276 |
| `show_morphemes` | 769,718 |
| Total tokens (sum of counts) | ~10.7 million |

Last show inserted: **LUPIN THE IIIRD. Mine Fujiko no Uso** (anime_movie category).

### Resuming the parse run

```sh
python3.12 tools/subParser/parse.py subs/minSubs/ -o subs/japaneseShowGraph.db --resume
```

The `--resume` flag skips any show whose (title, category) pair is already in the DB.

## Database Schema

```sql
-- Unique morphemes (surface + dictionary form + POS + reading)
CREATE TABLE morphemes (
    id INTEGER PRIMARY KEY,
    surface_form TEXT NOT NULL,
    dictionary_form TEXT NOT NULL,
    part_of_speech TEXT NOT NULL,
    reading TEXT,
    is_oov INTEGER NOT NULL DEFAULT 0,  -- 1 if not in SudachiPy dictionary
    UNIQUE(surface_form, dictionary_form, part_of_speech, reading)
);

-- One row per show (title + category is unique)
CREATE TABLE shows (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT,
    episode_count INTEGER,
    UNIQUE(title, category)
);

-- Morpheme frequency per show
CREATE TABLE show_morphemes (
    show_id INTEGER NOT NULL,
    morpheme_id INTEGER NOT NULL,
    count INTEGER NOT NULL,
    PRIMARY KEY (show_id, morpheme_id),
    FOREIGN KEY (show_id) REFERENCES shows(id),
    FOREIGN KEY (morpheme_id) REFERENCES morphemes(id)
);

-- For TF-IDF lookups
CREATE INDEX idx_show_morphemes_morpheme ON show_morphemes(morpheme_id);
```

### Columns

| Table | Column | Description |
|-------|--------|-------------|
| `morphemes` | `surface_form` | Token as it appears in text |
| `morphemes` | `dictionary_form` | Base/lemma form |
| `morphemes` | `part_of_speech` | Full POS tag (e.g. `名詞-普通名詞-一般`) |
| `morphemes` | `reading` | Katakana reading from SudachiPy |
| `morphemes` | `is_oov` | `1` if the token was not found in the SudachiPy dictionary (out-of-vocabulary) |
| `shows` | `title` | Show directory name from kitsunekko |
| `shows` | `category` | One of `anime_movie`, `anime_tv`, `drama_movie`, `drama_tv`, `unsorted` |
| `shows` | `episode_count` | Number of subtitle files found for this show |
| `show_morphemes` | `count` | How many times this morpheme appeared across all episodes of the show |

## Tool Reference

### `tools/subMinifier/minify.py`

Strips formatting/tags from .srt, .ass, and .ssa files. Outputs one sentence per line.

```sh
python minify.py path/to/subs/ -o path/to/clean_subs/
```

### `tools/subParser/parse.py`

Tokenizes minified text with SudachiPy and populates the SQLite database.

```sh
python3.12 tools/subParser/parse.py subs/minSubs/ -o subs/japaneseShowGraph.db           # full run
python3.12 tools/subParser/parse.py subs/minSubs/ -o subs/japaneseShowGraph.db --resume   # resume
python3.12 tools/subParser/parse.py subs/minSubs/ -o subs/japaneseShowGraph.db --limit 5  # test
python3.12 tools/subParser/parse.py subs/minSubs/ -o subs/japaneseShowGraph.db --mode C   # longest segmentation
```

- **`--mode A/B/C`** — SudachiPy segmentation mode (default: B, balanced)
- **`--resume`** — skip shows already in the DB
- **`--limit N`** — process only the first N shows
- **`--min-kana-ratio F`** — minimum kana token ratio to accept a show (default: 0.1)
- **`--min-tokens N`** — minimum total tokens to accept a show (default: 50)

**Filtering pipeline:**

1. Non-Japanese filename filtering — skips files with language codes (`.en.`, `[eng]`, `.zh.`, etc.)
2. Line-level — drops lines without Japanese characters (hiragana/katakana/kanji)
3. POS at tokenization — discards `補助記号` (supplementary symbols) and `空白` (whitespace)
4. OOV flagging — marks morphemes not in the SudachiPy dictionary (`is_oov = 1`)
5. Show-level validation — rejects shows below kana ratio or token count thresholds

### `tools/grapher/graph.py`

Reads the SQLite database and produces a two-level graph exported as GraphML files.

```sh
python3.12 tools/grapher/graph.py subs/japaneseShowGraph.db -o output/                           # defaults
python3.12 tools/grapher/graph.py subs/japaneseShowGraph.db -o output/ --topk 20 --threshold 0.01  # tune similarity sparsity
python3.12 tools/grapher/graph.py subs/japaneseShowGraph.db -o output/ --resolution 1.0            # tune cluster granularity
python3.12 tools/grapher/graph.py subs/japaneseShowGraph.db -o output/ --threads 4                 # parallel similarity
python3.12 tools/grapher/graph.py subs/japaneseShowGraph.db -o output/ --names output/names.json   # custom cluster names
```

- **`--topk N`** — keep top-N most similar neighbours per show (default: 20)
- **`--threshold F`** — minimum cosine similarity to retain (default: 0.01)
- **`--resolution F`** — Leiden resolution; higher = more clusters (default: 1.0)
- **`--threads N`** — threads for sparse matrix multiplication (default: 1)
- **`--min-tokens N`** — minimum total tokens to include a show (default: 500)
- **`--min-kana-ratio F`** — minimum kana token ratio to include a show (default: 0.1)
- **`--plot`** — generate PNG visualizations alongside GraphML files
- **`--names PATH`** — JSON file mapping cluster IDs to human-readable names

**Filtering pipeline (on top of parser-side filtering):**

1. Morpheme POS exclusion — `名詞-固有名詞` (proper nouns), `名詞-数詞` (numerals), `記号` (symbols), `感動詞` (interjections)
2. OOV exclusion — morphemes flagged `is_oov` by the parser are dropped from TF-IDF vectors
3. Show-level quality gate — stricter than the parser: `--min-tokens 500` (vs parser's 50), `--min-kana-ratio 0.1`

Output: `cluster_graph.graphml` (domain-level), `cluster_N.graphml` (per-cluster show graphs), `full_graph.json` (complete adjacency), `full_layout.json` (precomputed 2D positions), optionally `names.json` (cluster labels)

## Dependencies

- Python 3.12 (SudachiPy has no prebuilt wheel for Python 3.14)
- `sudachipy` and `sudachidict_core` — installed via `pip3.12 install --break-system-packages sudachipy sudachidict_core`
- `scipy`, `scikit-learn`, `sparse-dot-topn`, `igraph`, `leidenalg`, `networkx` — installed via `pip3.12 install --break-system-packages scipy scikit-learn sparse-dot-topn igraph leidenalg networkx`

## What's Next

1. Finish the parse run (`--resume`)
2. Re-run the grapher on the complete database
3. Visualize in Gephi and iterate on parameters
4. Build learner traversal / recommendation layer
