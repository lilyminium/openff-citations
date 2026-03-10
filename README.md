# OpenFF Citations

An automated pipeline that maps the scientific literature around the [Open Force Field Initiative](https://openforcefield.org/). It fetches papers from [OpenAlex](https://openalex.org), embeds and clusters them by topic, and produces an interactive visualisation deployed to GitHub Pages.

## How it works

```
OpenFF publications & Zenodo DOIs
          │
          ▼
    Resolve → OpenAlex IDs
          │
          ▼
  Fetch citing papers (hop-1, hop-2)
          │
          ▼
  Embed with all-MiniLM-L6-v2
  (title + abstract + keywords × 10 + authors)
          │
          ▼
  UMAP 2D layout  ──fit on hop-0+1──▶  project hop-2
          │
          ▼
  HDBSCAN clustering (hop-0+1 only)
          │
          ▼
  TF-IDF lift labelling
  (Gensim phrase detection → centroid-weighted lift scoring)
          │
          ▼
  Interactive Plotly HTML → GitHub Pages
```

### Citation hops

| Hop | Papers | Description |
|-----|--------|-------------|
| 0 | ~250 | OpenFF publications and Zenodo software releases |
| 1 | ~800 | Papers that directly cite an OpenFF work |
| 2 | ~14 000 (subsampled) | Papers that cite a hop-1 paper |

Hop-2 papers are shown as a toggleable layer (off by default) to keep the map readable.

### Embedding

Each paper is embedded as: `title + abstract + keywords × 10 + author names` using [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384-dim, Apache 2.0). Keywords are repeated to weight topic signals more heavily than free text.

### Layout

UMAP is fit on the ~1 000 hop-0+1 papers only (`fit_transform`), keeping the map centred on OpenFF-relevant structure. Hop-2 papers are then projected into the same space via UMAP's approximate `transform()`.

### Clustering

HDBSCAN is run on the 2D UMAP coordinates of hop-0+1 papers (no cluster count needed). Low-density points are labelled "Uncategorised".

### Cluster labels

The label text is first processed with **Gensim Phrases** (bigram → trigram) to join collocations into single tokens ("molecular_dynamics", "neural_network_potentials"). TF-IDF lift scoring then ranks terms by how distinctively they appear in each cluster relative to the rest of the corpus, weighted by each document's proximity to its cluster centroid. Author names, URLs, and generic scientific hedging words ("proposed", "results", "method", …) are excluded. The top 3 terms form the cluster label.

## Output

`outputs/citations.html` — a self-contained interactive scatter plot.

- **Hover** a point to see title, authors, year, DOI, and keywords
- **Click** a point to open its DOI in a new tab
- **Legend** — single-click to toggle a cluster; double-click to isolate it
- **Hop-2 toggle** — checkbox in the control box (bottom-right)
- **Dark / light mode** — button in the control box

## Usage

```bash
# Recommended: use the cached corpus (skips OpenAlex fetch)
python pipeline.py \
  --publications inputs/publications.csv \
  --zenodo       inputs/zenodo.csv \
  --email        your@email.org

# Force a fresh fetch from OpenAlex
python pipeline.py \
  --publications inputs/publications.csv \
  --zenodo       inputs/zenodo.csv \
  --email        your@email.org \
  --no-cache

# Optional flags
#   --min-cluster-size INT   HDBSCAN min_cluster_size (default 60)
#   --min-samples INT        HDBSCAN min_samples (default 5)
#   --max-hop2 INT           cap on hop-2 papers kept (default 5000)
```

### Environment

```bash
mamba create -n openff-citations python=3.11
mamba activate openff-citations
pip install -r requirements.txt
```

## Automated updates

A [GitHub Actions workflow](.github/workflows/update-citations.yml) runs on the 1st of every month and on manual dispatch. It:

1. Checks out this repo and [`lilyminium/openff-stats`](https://github.com/lilyminium/openff-stats) (source of truth for `publications.csv` and `zenodo.csv`)
2. Runs the pipeline with `--no-cache` (fresh OpenAlex fetch)
3. Deploys `outputs/citations.html` to GitHub Pages as `index.html`

The workflow requires an `OPENALEX_EMAIL` repository secret (used as a polite contact in OpenAlex API requests).

## Repository layout

```
pipeline.py          Main pipeline script
requirements.txt     Python dependencies
inputs/              Local copies of input CSVs (used when running locally)
data/                Cached corpus JSON, embeddings, UMAP coords
outputs/             Generated HTML visualisation
.github/workflows/   GitHub Actions workflow
```

## Data source

All citation data is fetched from [OpenAlex](https://openalex.org) — a free, open scholarly graph of ~250 M works. Coverage is estimated at ~80–85%; some citation edges and abstracts may be missing.
