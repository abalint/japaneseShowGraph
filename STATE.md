# Japanese Show Graph — Project State

## Overview

Build a graph/similarity database of Japanese TV shows and movies based on morphological analysis of their subtitle files. Raw subtitles are sourced from a kitsunekko mirror, cleaned into plain text, then parsed with SudachiPy into morpheme frequency tables in SQLite. TF-IDF vectors and show-to-show cosine similarity are computed from the database, then clustered into vocabulary domains and exported as GraphML graphs for Gephi.

## Pipeline

| Step | Tool | Status |
|------|------|--------|
| 0. Source data | `subs/kitsunekko-mirror-main/` (13 GB raw .srt/.ass/.ssa) | Done |
| 1. Minify subtitles | `tools/subMinifier/minify.py` → `subs/minSubs/` (5.3 GB clean .txt) | Done |
| 2. Parse & insert into DB | `tools/subParser/parse.py` → `subs/japaneseShowGraph.db` | **In progress** |
| 3. TF-IDF + similarity + graph | `tools/grapher/graph.py` → `output/*.graphml` | Done |

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
├── output/                             # GraphML graph files for Gephi
│   ├── cluster_graph.graphml              # domain-level graph (clusters as nodes)
│   └── cluster_N.graphml                  # per-cluster show graphs
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

### `tools/grapher/graph.py`

Reads the SQLite database and produces a two-level graph exported as GraphML files.

```sh
python3.12 tools/grapher/graph.py subs/japaneseShowGraph.db -o output/                           # defaults
python3.12 tools/grapher/graph.py subs/japaneseShowGraph.db -o output/ --topk 20 --threshold 0.01  # tune similarity sparsity
python3.12 tools/grapher/graph.py subs/japaneseShowGraph.db -o output/ --resolution 1.0            # tune cluster granularity
python3.12 tools/grapher/graph.py subs/japaneseShowGraph.db -o output/ --threads 4                 # parallel similarity
```

- **`--topk N`** — keep top-N most similar neighbours per show (default: 20)
- **`--threshold F`** — minimum cosine similarity to retain (default: 0.01)
- **`--resolution F`** — Leiden resolution; higher = more clusters (default: 1.0)
- **`--threads N`** — threads for sparse matrix multiplication (default: 1)

Output: `cluster_graph.graphml` (domain-level) + `cluster_N.graphml` (per-cluster show graphs)

## Dependencies

- Python 3.12 (SudachiPy has no prebuilt wheel for Python 3.14)
- `sudachipy` and `sudachidict_core` — installed via `pip3.12 install --break-system-packages sudachipy sudachidict_core`
- `scipy`, `scikit-learn`, `sparse-dot-topn`, `igraph`, `leidenalg`, `networkx` — installed via `pip3.12 install --break-system-packages scipy scikit-learn sparse-dot-topn igraph leidenalg networkx`

## What's Next

1. Finish the parse run (`--resume`)
2. Re-run the grapher on the complete database
3. Visualize in Gephi and iterate on parameters
4. Build learner traversal / recommendation layer
