#!/usr/bin/env python3
"""
OpenFF Citation Analysis Pipeline
==================================
Builds a citation neighbourhood graph centred on OpenFF papers.

Corpus structure
----------------
  hop 0 : OpenFF papers themselves (from publications.csv and zenodo.csv)
  hop 1 : papers that cite any hop-0 paper
  hop 2 : papers that cite any hop-1 paper
  …      (configurable via --n-steps, default=2)

Edges are drawn between papers across consecutive hops, showing the
citation flow from the wider literature into the OpenFF ecosystem.

Usage
-----
    python pipeline.py                          # hops 0-2 (default)
    python pipeline.py --n-steps 1             # hops 0-1 only
    python pipeline.py --no-cache              # ignore cached data
    python pipeline.py --email you@example.com # higher OpenAlex rate limit
    python pipeline.py --output my.html
"""

import argparse
import colorsys
import json
import random
import re
import time
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent
DATA_DIR    = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"

DEFAULT_PUBLICATIONS = REPO_ROOT / "inputs" / "publications.csv"
DEFAULT_ZENODO       = REPO_ROOT / "inputs" / "zenodo.csv"

OPENALEX_BASE = "https://api.openalex.org"
BATCH_SIZE = 50   # IDs per `cites:` batch request
REFS_BATCH = 50   # IDs per `ids.openalex:` batch (URL length limit ~600 chars)
CACHE_VERSION = "v4"   # bump when cache format changes

_session = requests.Session()

_TYPE_LABELS = {
    "article":             "Article",
    "review":              "Review",
    "preprint":            "Preprint",
    "book-chapter":        "Book Chapter",
    "book":                "Book",
    "proceedings-article": "Conference Paper",
    "editorial":           "Editorial",
    "letter":              "Letter",
    "dataset":             "Dataset",
    "software":            "Software",
    "dissertation":        "Thesis",
    "other":               "Other",
    "peer-review":         "Peer Review",
    "reference-entry":     "Reference Entry",
    "report":              "Report",
    "retraction":          "Retraction",
}

# How many extra times keywords are repeated in TF-IDF label text vs title+abstract.
KEYWORD_REPEAT = 5
# How many extra times keywords are repeated in embedding text (higher = stronger
# topic signal for clustering).
KEYWORD_REPEAT_EMBED = 10

# Generic scientific/English words that appear uniformly across all clusters and
# would make labels uninformative.  scikit-learn's ENGLISH_STOP_WORDS covers
# functional vocabulary only; this extends it with hedging verbs, method nouns,
# and placeholder-abstract tokens common in scholarly text.
_SCIENCE_STOPS: frozenset[str] = frozenset({
    # placeholder / retrieval artefacts
    "unable", "retrieve", "retrieved", "retrieval", "preview", "download",
    "abstract", "preprint", "doi", "fig", "figure", "table", "eq", "al",
    # generic method / result verbs
    "propose", "proposed", "present", "presented", "show", "shown", "shows",
    "demonstrate", "demonstrated", "demonstrates", "report", "reported", "reports",
    "develop", "developed", "develops", "introduce", "introduced",
    "investigate", "investigated", "describe", "described",
    "evaluate", "evaluated", "assess", "assessed", "validate", "validated",
    "apply", "applied", "compare", "compared", "achieve", "achieved",
    "obtain", "obtained", "improve", "improved", "calculate", "calculated",
    "predict", "predicted", "require", "required",
    "use", "used", "using", "based", "provide", "provided",
    # generic quantifiers / adjectives
    "new", "novel", "different", "various", "specific", "particular",
    "high", "low", "large", "small", "significant", "important",
    "efficient", "effective", "accurate", "recent", "current", "existing",
    "previous", "computational", "experimental", "theoretical",
    # generic nouns
    "paper", "papers", "study", "studies", "work", "works",
    "method", "methods", "approach", "approaches", "result", "results",
    "analysis", "model", "models", "performance", "system", "systems",
    "problem", "problems", "challenge", "challenges",
    "prediction", "predictions", "calculation", "calculations",
    "improvement", "improvements", "accuracy",
    # numbers as words
    "one", "two", "three", "four", "five",
})

_DISAMBIGUATION_RE = re.compile(r'\s*\([^)]+\)\s*$')


def _extract_kw_terms(work: dict, min_score: float = 0.4) -> list[str]:
    """Return cleaned, deduplicated keyword terms from an OpenAlex work dict.

    Uses `concepts` (the best-covered field) filtered to score ≥ min_score and
    strips Wikipedia disambiguation suffixes like "(fiction)" or "(mathematics)".
    High-quality `keywords` and `topics` terms are appended after.
    """
    seen: dict[str, None] = {}

    def _add(name: str) -> None:
        clean = _DISAMBIGUATION_RE.sub("", name).strip()
        if clean and clean not in seen:
            seen[clean] = None

    for c in (work.get("concepts") or []):
        if (c.get("score") or 0) >= min_score:
            _add(c.get("display_name", ""))

    for k in (work.get("keywords") or []):
        if (k.get("score") or 0) >= min_score:
            _add(k.get("display_name", ""))

    for t in (work.get("topics") or []):
        if (t.get("score") or 0) >= min_score:
            _add(t.get("display_name", ""))
            for sub in ("subfield", "field"):
                _add((t.get(sub) or {}).get("display_name", ""))

    return list(seen)


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAlex API helpers
# ═══════════════════════════════════════════════════════════════════════════════

def oa_get(url: str, params: dict = None, email: str = "", retries: int = 4) -> dict | None:
    p = {**(params or {})}
    if email:
        p["mailto"] = email
    for attempt in range(retries):
        try:
            r = _session.get(url, params=p, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(5 * 2 ** attempt)
            elif r.status_code == 404:
                return None
            else:
                time.sleep(1)
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def fetch_pages(filter_str: str, select: str, email: str = "") -> list[dict]:
    """Fetch all cursor-paginated results for a /works filter."""
    results, cursor = [], "*"
    while cursor:
        data = oa_get(
            f"{OPENALEX_BASE}/works",
            params={"filter": filter_str, "per-page": 200, "cursor": cursor, "select": select},
            email=email,
        )
        if not data:
            break
        batch = data.get("results", [])
        results.extend(batch)
        cursor = data.get("meta", {}).get("next_cursor") if batch else None
        time.sleep(0.11)
    return results


def resolve_doi(doi: str, email: str = "") -> dict | None:
    doi_clean = doi.strip().lstrip("https://doi.org/").lstrip("http://doi.org/")
    return oa_get(f"{OPENALEX_BASE}/works/doi:{doi_clean}", email=email)


# ═══════════════════════════════════════════════════════════════════════════════
# Source loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_openff_sources(publications_csv: Path, zenodo_csv: Path) -> dict[str, str]:
    """Return {doi_lower -> title} for all OpenFF pubs and Zenodo records."""
    sources: dict[str, str] = {}
    for _, row in pd.read_csv(publications_csv).iterrows():
        doi = str(row.get("DOI", "")).strip().lower()
        if doi and doi != "nan":
            sources[doi] = str(row.get("title", "")).strip()
    for _, row in pd.read_csv(zenodo_csv).iterrows():
        doi   = str(row.get("doi",   "")).strip().lower()
        title = str(row.get("title", "")).strip()
        rtype = str(row.get("resource_type", "other")).strip()
        if doi and doi != "nan":
            sources[doi] = f"[{rtype}] {title}"
    return sources


# ═══════════════════════════════════════════════════════════════════════════════
# Citation + reference fetching
# ═══════════════════════════════════════════════════════════════════════════════

# Fields we want for every work — type added for article-type classification
_SELECT = (
    "id,doi,title,publication_year,authorships,"
    "abstract_inverted_index,cited_by_count,referenced_works,primary_topic,type,"
    "keywords,concepts,topics"
)


def resolve_openff_works(sources: dict[str, str], email: str = "") -> dict[str, dict]:
    """Resolve OpenFF DOIs → {oa_short_id: work_dict}."""
    resolved: dict[str, dict] = {}
    print(f"\n=== Resolving {len(sources)} OpenFF DOIs in OpenAlex ===")
    for doi, title in tqdm(sources.items(), desc="Resolving"):
        work = resolve_doi(doi, email=email)
        if work and "id" in work:
            oa_id = work["id"].split("/")[-1]
            work["_source_doi"]   = doi
            work["_source_title"] = work.get("title") or title
            resolved[oa_id] = work
        time.sleep(0.12)
    print(f"  → {len(resolved)}/{len(sources)} resolved")
    return resolved


def batch_fetch_citing(oa_ids: list[str], desc: str, email: str = "") -> list[dict]:
    """Batch-fetch all works citing any of the given OA IDs using `cites:id1|id2|…`."""
    all_works: list[dict] = []
    ids = list(oa_ids)
    for i in tqdm(range(0, len(ids), BATCH_SIZE), desc=desc):
        batch = ids[i : i + BATCH_SIZE]
        works = fetch_pages("cites:" + "|".join(batch), _SELECT, email=email)
        all_works.extend(works)
    return all_works


def batch_fetch_by_id(oa_ids: list[str], desc: str, email: str = "") -> list[dict]:
    """Fetch full metadata for a list of OA IDs using `ids.openalex:id1|id2|…`."""
    all_works: list[dict] = []
    ids = list(oa_ids)
    for i in tqdm(range(0, len(ids), REFS_BATCH), desc=desc):
        batch = ids[i : i + REFS_BATCH]
        filter_str = "ids.openalex:" + "|".join(batch)
        works = fetch_pages(filter_str, _SELECT, email=email)
        all_works.extend(works)
    return all_works


def collect_corpus(
    sources: dict[str, str],
    n_steps: int = 2,
    email: str = "",
    cache_path: Path | None = None,
    force_refresh: bool = False,
) -> tuple[dict, dict, dict, list, dict]:
    """
    Build the citation neighbourhood corpus.

    Parameters
    ----------
    force_refresh : if True, ignore any existing cache on load but still save after fetching.

    Returns
    -------
    papers       : {oa_short_id -> work_dict}
    cites_openff : {oa_short_id -> [openff_doi, …]}
    roles        : {oa_short_id -> hop_int}   (0=OpenFF, 1=direct citers, …)
    edges        : [[citing_id, cited_id], …]  (between adjacent hops)
    openff_works : {oa_short_id -> work_dict}
    """
    if cache_path and cache_path.exists() and not force_refresh:
        print(f"\nLoading cached corpus from {cache_path}")
        with open(cache_path) as f:
            d = json.load(f)
        return d["papers"], d["cites_openff"], d["roles"], d["edges"], d["openff_works"]

    papers: dict[str, dict]  = {}
    roles:  dict[str, int]   = {}
    edges:  list[list[str]]  = []

    # ── 1. Resolve OpenFF DOIs → OpenAlex works ────────────────────────────────
    openff_works = resolve_openff_works(sources, email=email)
    openff_id_to_doi = {oid: w["_source_doi"] for oid, w in openff_works.items()}
    hop0_ids = set(openff_works.keys())

    # ── 2. Hop 0: fetch full metadata for OpenFF papers ────────────────────────
    print(f"\n=== Hop 0: {len(hop0_ids)} OpenFF papers ===")
    hop0_meta = batch_fetch_by_id(list(hop0_ids), "  Fetching hop-0", email=email)
    for work in hop0_meta:
        oa_id = work["id"].split("/")[-1]
        papers[oa_id] = work
        roles[oa_id]  = 0
    for oa_id, work in openff_works.items():
        if oa_id not in papers:
            papers[oa_id] = work
            roles[oa_id]  = 0

    # Intra-hop-0 edges (OpenFF papers citing each other)
    for oa_id, work in papers.items():
        if roles[oa_id] != 0:
            continue
        refs = {r.split("/")[-1] for r in (work.get("referenced_works") or [])}
        for rid in refs:
            if rid in hop0_ids and rid != oa_id:
                edges.append([oa_id, rid])

    print(f"  → {sum(1 for r in roles.values() if r == 0)} hop-0 papers in corpus")

    # ── 3. Citation hops 1 … n_steps ──────────────────────────────────────────
    current_hop_ids = list(roles.keys())

    for hop in range(1, n_steps + 1):
        if not current_hop_ids:
            break
        prev_hop_set = set(current_hop_ids)
        print(f"\n=== Hop {hop}: papers citing {len(current_hop_ids)} hop-{hop-1} papers ===")
        hop_raw = batch_fetch_citing(current_hop_ids, f"  Fetching hop-{hop}", email=email)

        new_ids: list[str] = []
        for work in hop_raw:
            oa_id = work["id"].split("/")[-1]
            refs  = {r.split("/")[-1] for r in (work.get("referenced_works") or [])}
            for rid in refs:
                if rid in prev_hop_set:
                    edges.append([oa_id, rid])
            if oa_id in papers:
                continue
            papers[oa_id] = work
            roles[oa_id]  = hop
            new_ids.append(oa_id)

        current_hop_ids = new_ids
        print(f"  → {len(new_ids)} new papers at hop {hop}")

    # ── 4. Build cites_openff ──────────────────────────────────────────────────
    cites_openff: dict[str, list] = {}
    for oa_id, work in papers.items():
        refs  = {r.split("/")[-1] for r in (work.get("referenced_works") or [])}
        cited = [openff_id_to_doi[r] for r in refs if r in openff_id_to_doi]
        cites_openff[oa_id] = cited

    edges = [list(e) for e in {tuple(e) for e in edges}]

    print(f"\nCorpus total: {len(papers)} papers")
    hop_counts: dict[int, int] = {}
    for r in roles.values():
        hop_counts[r] = hop_counts.get(r, 0) + 1
    for h, c in sorted(hop_counts.items()):
        print(f"  hop-{h}: {c}")
    print(f"  edges: {len(edges)}")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(
                {"papers": papers, "cites_openff": cites_openff,
                 "roles": roles, "edges": edges, "openff_works": openff_works},
                f,
            )
        print(f"Saved cache → {cache_path}")

    return papers, cites_openff, roles, edges, openff_works


# ═══════════════════════════════════════════════════════════════════════════════
# DataFrame builder
# ═══════════════════════════════════════════════════════════════════════════════

def decode_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos_word: dict[int, str] = {}
    for word, positions in inv.items():
        for p in positions:
            pos_word[p] = word
    return " ".join(pos_word[i] for i in sorted(pos_word))


def format_authors(authorships: list, max_n: int = 5) -> str:
    names = []
    for a in (authorships or [])[:max_n]:
        name = (a.get("author") or {}).get("display_name", "")
        if name:
            names.append(name)
    extra = len(authorships or []) - max_n
    if extra > 0:
        names.append(f"+ {extra} more")
    return "; ".join(names)


def build_dataframe(
    papers: dict,
    cites_openff: dict,
    roles: dict,
    sources: dict,
) -> pd.DataFrame:
    rows = []
    for oa_id, work in papers.items():
        title = (work.get("title") or "").strip()
        if not title:
            continue

        abstract = decode_abstract(work.get("abstract_inverted_index"))
        authors  = format_authors(work.get("authorships", []))
        doi_raw  = work.get("doi", "") or ""
        doi      = doi_raw.replace("https://doi.org/", "").replace("http://doi.org/", "")

        cited_dois   = cites_openff.get(oa_id, [])
        # Store as semicolon-separated for CSV; build_hover will split + reformat
        cited_titles = "; ".join(sources.get(d, d)[:80] for d in cited_dois)

        raw_type     = (work.get("type") or "").lower()
        article_type = _TYPE_LABELS.get(raw_type, raw_type.title() if raw_type else "")

        text = f"{title}. {abstract}".strip() if abstract else title
        hop  = int(roles.get(oa_id, 1))

        kw_terms = _extract_kw_terms(work)
        keywords = "; ".join(kw_terms)   # semicolon-separated for display; ; → space in TF-IDF/embed

        rows.append({
            "oa_id":          oa_id,
            "title":          title,
            "abstract":       abstract,
            "year":           work.get("publication_year"),
            "doi":            doi,
            "doi_url":        f"https://doi.org/{doi}" if doi else "",
            "authors":        authors,
            "cited_by_count": int(work.get("cited_by_count") or 0),
            "hop":            hop,
            "article_type":   article_type,
            "cites_openff":   cited_titles,
            "text":           text,
            "keywords":       keywords,
        })

    df = pd.DataFrame(rows).drop_duplicates("oa_id").reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding, UMAP, clustering
# ═══════════════════════════════════════════════════════════════════════════════

def embed(df: pd.DataFrame, cache_path: Path | None = None, force_refresh: bool = False) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    if cache_path and cache_path.exists() and not force_refresh:
        cached = np.load(cache_path)
        if cached.shape[0] == len(df):
            print(f"\nLoading cached embeddings from {cache_path}")
            return cached
        print(f"\nCached embeddings shape mismatch ({cached.shape[0]} vs {len(df)} rows), recomputing…")

    print("\n=== Computing embeddings (all-MiniLM-L6-v2) ===")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def _embed_text(row: pd.Series) -> str:
        parts = [row.get("text") or ""]   # title + abstract
        kw = (row.get("keywords") or "").replace(";", " ").strip()
        if kw:
            parts.extend([kw] * KEYWORD_REPEAT_EMBED)   # weight keywords heavily
        raw_authors = row.get("authors") or ""
        author_clean = "; ".join(
            p.strip() for p in raw_authors.split(";")
            if p.strip() and not p.strip().startswith("+")
        )
        if author_clean:
            parts.append(author_clean)
        return " ".join(parts)

    texts = df.apply(_embed_text, axis=1).tolist()
    embs  = model.encode(texts, show_progress_bar=True, batch_size=64,
                         normalize_embeddings=True)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embs)
        print(f"Saved embeddings → {cache_path}")
    return embs


def run_umap(
    embs: np.ndarray,
    fit_mask: np.ndarray | None = None,
    cache_path: Path | None = None,
    seed: int = 42,
    force_refresh: bool = False,
) -> np.ndarray:
    """
    Reduce embeddings to 2D via UMAP.

    If `fit_mask` is provided (boolean array, same length as embs), the
    UMAP manifold is fit only on the True rows (hop-0+1 papers), then all
    remaining rows are projected via transform().  This keeps the layout
    centred on OpenFF-relevant structure while still placing hop-2 papers.
    """
    import umap

    if cache_path and cache_path.exists() and not force_refresh:
        cached = np.load(cache_path)
        if cached.shape[0] == embs.shape[0]:
            print(f"\nLoading cached UMAP coords from {cache_path}")
            return cached
        print(f"\nCached UMAP shape mismatch, recomputing…")

    print("\n=== UMAP → 2D ===")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.05,
        metric="cosine",
        random_state=seed,
        low_memory=False,
    )

    if fit_mask is not None and not fit_mask.all():
        n_fit = int(fit_mask.sum())
        n_transform = int((~fit_mask).sum())
        print(f"  Fitting on {n_fit} hop-0+1 papers; transforming {n_transform} hop-2+ papers…")
        coords = np.zeros((len(embs), 2))
        coords[fit_mask]  = reducer.fit_transform(embs[fit_mask])
        coords[~fit_mask] = reducer.transform(embs[~fit_mask])
    else:
        coords = reducer.fit_transform(embs)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, coords)
        print(f"Saved UMAP coords → {cache_path}")
    return coords


def cluster_hdbscan(
    coords: np.ndarray,
    min_cluster_size: int = 60,
    min_samples: int = 5,
) -> np.ndarray:
    import hdbscan

    print("\n=== Clustering (HDBSCAN) ===")
    cl = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = cl.fit_predict(coords)
    n_cl = len(set(labels)) - (1 if -1 in labels else 0)
    n_no = int((labels == -1).sum())
    print(f"  → {n_cl} clusters | {n_no} uncategorised points")
    return labels


# ═══════════════════════════════════════════════════════════════════════════════
# Data-driven cluster labelling with TF-IDF
# ═══════════════════════════════════════════════════════════════════════════════

def label_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    coords_fit: np.ndarray,
    top_n: int = 3,
) -> dict[int, str]:
    """
    For each cluster find its most distinctive unigrams relative to the rest
    of the corpus using TF-IDF lift scoring. No predefined keyword map.

    Publication keywords (concepts/topics from OpenAlex) are appended to each
    document text KEYWORD_REPEAT extra times so they carry proportionally more
    weight in the TF-IDF matrix than the free-text title+abstract.

    Within each cluster, documents are weighted by proximity to the cluster
    centroid so that the most representative papers drive the label rather than
    peripheral members.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

    _URL_RE = re.compile(r'https?://\S+|www\.\S+')

    def _weighted_text(row: pd.Series) -> str:
        base = (row.get("text") or "").lower()
        kw_raw = (row.get("keywords") or "").lower()
        if kw_raw.strip():
            # Repeat each keyword individually, period-separated, so the
            # vectorizer cannot form n-grams spanning across repetitions
            # (e.g. "presentation presentation presentation presentation").
            kw_terms_list = [t.strip() for t in kw_raw.split(";") if t.strip()]
            repeated = ". ".join(kw_terms_list * KEYWORD_REPEAT)
            base = base + " " + repeated
        # Strip URLs so that path components (usernames, repo names, etc.)
        # never surface as cluster labels.
        base = _URL_RE.sub(" ", base)
        return base

    texts = df.apply(_weighted_text, axis=1).tolist()
    unique_ids = sorted(set(labels))

    # ── Phrase detection (Gensim) ────────────────────────────────────────────
    # Join statistically significant collocations into single underscore-joined
    # tokens before TF-IDF so that "molecular_dynamics" and
    # "neural_network_potentials" are treated as atomic units rather than
    # being assembled (badly) from overlapping n-grams.
    print("  Detecting phrases (bigrams → trigrams) …")
    sentences = [t.split() for t in texts]
    bigram_model  = Phrases(sentences,  min_count=5, threshold=10,
                            connector_words=ENGLISH_CONNECTOR_WORDS)
    trigram_model = Phrases(bigram_model[sentences], min_count=3, threshold=8,
                            connector_words=ENGLISH_CONNECTOR_WORDS)
    texts_phrased = [" ".join(trigram_model[bigram_model[s]]) for s in sentences]

    # Build author-name stop words: split every author token and blacklist it so
    # that individual names (e.g. "Matt", "Thompson") never appear as labels.
    author_stops: set[str] = set()
    for raw in df["authors"].dropna():
        for part in raw.split(";"):
            part = part.strip()
            if part.startswith("+"):   # "+ N more"
                continue
            for tok in part.split():
                tok = tok.strip(".,").lower()
                if len(tok) > 1 and tok.isalpha():
                    author_stops.add(tok)
    stop_words = list(ENGLISH_STOP_WORDS | author_stops | _SCIENCE_STOPS)

    print("\n=== Labelling clusters (TF-IDF) ===")
    global_vec = TfidfVectorizer(
        # ngram_range=(1,1): phrases are already joined by Gensim, so unigrams
        # here correspond to multi-word scientific terms like molecular_dynamics.
        ngram_range=(1, 1),
        stop_words=stop_words,
        max_features=30_000,
        min_df=3,
        sublinear_tf=True,
        token_pattern=r"[a-z][a-z_]+[a-z]",  # allow underscores inside tokens
    )
    global_vec.fit(texts_phrased)
    feature_names = global_vec.get_feature_names_out()

    # Global mean TF-IDF used as denominator smoothing.
    # Without it, terms absent from the rest get lift → ∞ even if they appear
    # in only 1–2 cluster documents, producing spurious label terms.
    X_global    = global_vec.transform(texts_phrased)
    global_mean = np.asarray(X_global.mean(axis=0)).ravel()

    cluster_names: dict[int, str] = {}

    for cid in unique_ids:
        if cid == -1:
            cluster_names[-1] = "Uncategorised"
            continue

        mask        = labels == cid
        cluster_idx = np.where(mask)[0]
        rest_idx    = np.where(~mask)[0]

        cluster_texts = [texts_phrased[i] for i in cluster_idx]
        rest_texts    = [texts_phrased[i] for i in rest_idx]

        X_cl   = global_vec.transform(cluster_texts)
        X_rest = global_vec.transform(rest_texts)

        # Weight each cluster document by proximity to the cluster centroid
        # so that the most representative papers drive the label rather than
        # peripheral or noisy members.
        c_coords = coords_fit[cluster_idx]
        centroid  = c_coords.mean(axis=0)
        dists     = np.linalg.norm(c_coords - centroid, axis=1)
        weights   = 1.0 / (dists + 1e-3)
        weights  /= weights.sum()

        mean_cl   = np.asarray(X_cl.T.dot(weights)).ravel()
        mean_rest = np.asarray(X_rest.mean(axis=0)).ravel()

        # Smoothed lift: denominator includes a fraction of the global mean so
        # that terms absent from the rest (mean_rest≈0) cannot achieve infinite
        # lift from a single occurrence.  log1p further compresses extreme lift
        # values, biasing labels toward terms that are densely present in the
        # cluster rather than merely absent everywhere else.
        lift  = mean_cl / (mean_rest + 0.1 * global_mean + 1e-12)
        score = mean_cl * np.log1p(lift)

        top_indices = score.argsort()[::-1][:50]   # candidate pool
        chosen = []
        for idx in top_indices:
            term = feature_names[idx]
            # Skip tokens that are pure repetitions of one word (shouldn't
            # occur now that Gensim handles phrasing, but kept as a guard).
            toks = term.split("_")
            if len(toks) > 1 and len(set(toks)) == 1:
                continue
            if not any(term in prev or prev in term for prev in chosen):
                chosen.append(term)
            if len(chosen) >= top_n:
                break

        cluster_names[cid] = " · ".join(t.replace("_", " ").title() for t in chosen)

    for cid, name in sorted(cluster_names.items()):
        count = int((labels == cid).sum())
        print(f"  {cid:3d}  ({count:4d} papers)  {name}")

    return cluster_names


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

_PALETTE_BASE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
    "#AEC7E8", "#FFBB78", "#98DF8A", "#FF9896", "#C5B0D5",
    "#C49C94", "#F7B6D2", "#C7C7C7", "#DBDB8D", "#9EDAE5",
]


def _semantic_color_map(
    cluster_names: dict[int, str],
    cluster_labels: np.ndarray,
    coords: np.ndarray,
) -> dict[str, str]:
    """
    Assign colors to cluster names so that spatially adjacent clusters have
    similar hues.

    Method: compute each cluster's centroid in 2D UMAP space, sort clusters
    by their angle from the global centroid, and map that rank linearly to hue
    (0–1 on the HSV wheel).  Because UMAP places semantically similar papers
    near each other, topically related clusters end up with similar colors.
    The cyclic nature of hue matches the cyclic nature of angular position.
    """
    uncategorised = "Uncategorised"

    # Compute 2D centroid for each named cluster
    centroids: dict[int, np.ndarray] = {}
    for cid in cluster_names:
        if cid == -1:
            continue
        mask = cluster_labels == cid
        if mask.any():
            centroids[cid] = coords[mask].mean(axis=0)

    if not centroids:
        return {uncategorised: "#444444"}

    all_cents = np.stack(list(centroids.values()))
    gc = all_cents.mean(axis=0)   # global center of cluster centroids

    # Angle of each cluster centroid from global center → hue
    angles = {
        cid: float(np.arctan2(cent[1] - gc[1], cent[0] - gc[0]))
        for cid, cent in centroids.items()
    }
    sorted_cids = sorted(centroids.keys(), key=lambda c: angles[c])
    n = len(sorted_cids)

    color_map = {uncategorised: "#444444"}
    for rank, cid in enumerate(sorted_cids):
        hue = rank / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.88)
        color_map[cluster_names[cid]] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    return color_map

# hop → (opacity, size_boost, border_width, border_color, symbol)
# "circle-open" renders as an outlined hollow circle — closest Plotly analog to a dashed outline
HOP_STYLE: dict[int, tuple] = {
    0: (1.00, 12, 2.5, "white",                    "square"),
    1: (0.85,  2, 1.0, "rgba(255,255,255,0.6)",     "circle"),
    2: (0.70,  0, 1.5, "rgba(255,255,255,0.5)",     "circle-open"),
    3: (0.50,  0, 1.0, "rgba(255,255,255,0.3)",     "circle-open"),
}
HOP_LABELS: dict[int, str] = {
    0: "OpenFF paper (hop 0)",
    1: "Cites OpenFF (hop 1)",
    2: "Cites hop-1 (hop 2)",
    3: "Cites hop-2 (hop 3)",
}

# Article type → Plotly marker symbol (hop-0 always uses "square" regardless)
_ARTICLE_TYPE_SYMBOL: dict[str, str] = {
    "Article":          "circle",
    "Review":           "diamond",
    "Preprint":         "triangle-up",
    "Conference Paper": "cross",
    "Software":         "star",
    "Dataset":          "star",
}
_DEFAULT_SYMBOL = "circle"

# Legend entries shown in "Article type" group
_TYPE_LEGEND: list[tuple[str, str]] = [
    ("circle",       "Article / Other"),
    ("diamond",      "Review"),
    ("triangle-up",  "Preprint"),
    ("cross",        "Conference paper"),
    ("star",         "Software / Dataset"),
]


def build_hover(row: pd.Series) -> str:
    doi_link = (
        f'<a href="{row.doi_url}" style="color:#7ec8e3" target="_blank">{row.doi}</a>'
        if row.doi_url else "N/A"
    )
    title_esc = str(row.title).replace("<", "&lt;").replace(">", "&gt;")
    year = str(int(row.year)) if pd.notna(row.year) else "?"
    hop  = int(row.hop)

    # Article type badge (only show when known)
    atype = str(row.get("article_type", "") or "")
    type_line = (
        f"<span style='color:#aaa; font-size:11px'>{atype}</span><br>"
        if atype else ""
    )

    # Cites OpenFF — one bullet per source on a new line
    if hop == 0:
        hop_line = "<b style='color:#ffd700'>OpenFF source paper</b>"
    elif hop == 1:
        raw = str(row.cites_openff or "")
        if raw:
            items = [s.strip() for s in raw.split(";") if s.strip()]
            if len(items) == 1:
                hop_line = f"<b>Cites OpenFF:</b> {items[0]}"
            else:
                bullets = "<br>&nbsp;&nbsp;• ".join(items)
                hop_line = f"<b>Cites OpenFF:</b><br>&nbsp;&nbsp;• {bullets}"
        else:
            hop_line = "<b>Cites OpenFF</b>"
    else:
        hop_line = f"<i>Hop {hop} — indirect citation</i>"

    kw_raw = str(row.get("keywords", "") or "")
    kw_line = ""
    if kw_raw.strip():
        kw_terms_disp = [t.strip() for t in kw_raw.split(";") if t.strip()][:8]
        kw_line = (
            f"<br><span style='color:#aaa;font-size:11px'>"
            f"<b>Keywords:</b> {', '.join(kw_terms_disp)}</span>"
        )

    return (
        f"{type_line}"
        f"<b>{title_esc[:150]}</b><br>"
        f"<span style='color:#ccc'>{row.authors[:120]}</span><br><br>"
        f"Year: {year} &nbsp;|&nbsp; Cited by: {row.cited_by_count}<br>"
        f"DOI: {doi_link}<br><br>"
        f"{hop_line}"
        f"{kw_line}"
    )


def create_viz(
    df: pd.DataFrame,
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_names: dict[int, str],
    edges: list,
    output_path: Path,
    max_edges: int = 10_000,
    metadata: dict | None = None,
) -> None:
    import plotly.graph_objects as go

    df = df.copy()
    df["x"]            = coords[:, 0]
    df["y"]            = coords[:, 1]
    df["cluster_id"]   = cluster_labels
    df["cluster_name"] = [cluster_names.get(c, f"Cluster {c}") for c in cluster_labels]
    df["hover"]        = df.apply(build_hover, axis=1)
    df["mbase"]        = np.clip(np.log1p(df["cited_by_count"]) * 2.5 + 4, 4, 22)

    id_to_xy  = dict(zip(df["oa_id"], zip(df["x"].tolist(), df["y"].tolist())))
    id_to_hop = dict(zip(df["oa_id"], df["hop"].tolist()))
    max_hop   = int(df["hop"].max())
    max_style = max(HOP_STYLE.keys())

    # ── Partition edges by hop transition ─────────────────────────────────────
    transition_edges: dict[tuple[int, int], list] = {}
    for src, tgt in edges:
        sh = id_to_hop.get(src)
        th = id_to_hop.get(tgt)
        if sh is None or th is None:
            continue
        lo, hi = min(sh, th), max(sh, th)
        if hi == lo + 1:
            transition_edges.setdefault((lo, hi), []).append((src, tgt))

    # Priority-weighted subsampling: keep ALL low-hop edges first, then
    # fill remaining budget with higher-hop edges.  This ensures that the
    # most informative connections (OpenFF ↔ direct citers) are never dropped.
    budget = max_edges
    for key in sorted(transition_edges.keys()):        # low hops first
        elist = transition_edges[key]
        if len(elist) <= budget:
            budget -= len(elist)
        else:
            transition_edges[key] = random.sample(elist, max(1, budget))
            budget = 0

    # ── Cluster colour map ────────────────────────────────────────────────────
    all_names     = sorted(df["cluster_name"].unique())
    uncategorised = "Uncategorised"
    ordered = [n for n in all_names if n != uncategorised]
    if uncategorised in all_names:
        ordered.append(uncategorised)

    color_map = _semantic_color_map(cluster_names, cluster_labels, coords)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = go.Figure()
    trace_idx        = 0
    hop_traces: dict[int, list[int]]       = {h: [] for h in range(max_hop + 1)}
    hop_point_traces: dict[int, list[int]] = {h: [] for h in range(max_hop + 1)}
    hop_legend_indices: list[int]          = []
    type_legend_indices: list[int]         = []
    edge_trace_map: dict[str, int]         = {}   # "lo-hi" -> trace index
    n_edges_drawn = 0

    # Edge traces — one per hop-transition; hop-2 edges start hidden
    for (lo, hi), elist in sorted(transition_edges.items()):
        ex, ey = [], []
        n = 0
        for src, tgt in elist:
            if src in id_to_xy and tgt in id_to_xy:
                sx, sy = id_to_xy[src]
                tx, ty = id_to_xy[tgt]
                ex += [sx, tx, None]
                ey += [sy, ty, None]
                n += 1
        if ex:
            fig.add_trace(go.Scatter(
                x=ex, y=ey,
                mode="lines",
                line=dict(width=0.4, color="rgba(200,200,200,0.15)"),
                hoverinfo="none",
                showlegend=False,
                visible=(hi < 2),   # hide hop-2 edges initially
                name=f"edges_{lo}_{hi}",
            ))
            edge_trace_map[f"{lo}-{hi}"] = trace_idx
            n_edges_drawn += n
            trace_idx += 1

    # Point traces — higher hops drawn first (behind), hop-0 last (on top)
    # For hop-0: always use "square" marker.
    # For hop-1+: per-point symbol encodes article type (circle/diamond/triangle/cross/star).
    hop_order = list(range(max_hop, -1, -1))
    for cname in ordered:
        for hop in hop_order:
            sub = df[(df["cluster_name"] == cname) & (df["hop"] == hop)]
            if sub.empty:
                continue
            sk = min(hop, max_style)
            opacity, size_boost, bw, bc, base_symbol = HOP_STYLE[sk]
            sizes = (sub["mbase"] + size_boost).tolist()
            show_legend = (hop == 1) and (cname != uncategorised)

            # Per-point symbol encodes article type for all hops.
            # Hop-0 is distinguished by thick white border + large size boost (HOP_STYLE).
            symbols = sub["article_type"].map(
                lambda t: _ARTICLE_TYPE_SYMBOL.get(t, _DEFAULT_SYMBOL)
            ).tolist()

            fig.add_trace(go.Scatter(
                x=sub["x"], y=sub["y"],
                mode="markers",
                name=cname,
                legendgroup=cname,
                showlegend=show_legend,
                visible=(hop < 2),   # hop-2 starts hidden
                text=sub["hover"],
                hovertemplate="%{text}<extra></extra>",
                customdata=sub["doi_url"],
                marker=dict(
                    size=sizes,
                    color=color_map.get(cname, "#888"),
                    opacity=opacity,
                    symbol=symbols,
                    line=dict(width=bw, color=bc),
                ),
            ))
            hop_traces[hop].append(trace_idx)
            hop_point_traces[hop].append(trace_idx)
            trace_idx += 1

    # Neighbourhood legend traces; hop-2 starts hidden to match point traces
    hop_counts = df.groupby("hop").size().to_dict()
    for hop in range(max_hop + 1):
        sk = min(hop, max_style)
        opacity, size_boost, bw, bc, symbol = HOP_STYLE[sk]
        label = HOP_LABELS.get(hop, f"Hop {hop}")
        cnt   = hop_counts.get(hop, 0)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            name=f"{label} ({cnt:,})",
            legendgroup="hops",
            legendgrouptitle_text="Neighbourhood",
            visible=(hop < 2),
            marker=dict(
                size=12, color="white", opacity=opacity,
                symbol=symbol,
                line=dict(width=bw, color="white"),
            ),
        ))
        hop_traces[hop].append(trace_idx)
        hop_legend_indices.append(trace_idx)
        trace_idx += 1

    # Article type legend traces (dummy points, always visible)
    for sym, label in _TYPE_LEGEND:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            name=label,
            legendgroup="article_types",
            legendgrouptitle_text="Article type",
            marker=dict(size=10, color="white", symbol=sym),
        ))
        type_legend_indices.append(trace_idx)
        trace_idx += 1

    # ── Layout ────────────────────────────────────────────────────────────────
    hop_parts  = [f"hop-{h}: {hop_counts[h]:,}" for h in sorted(hop_counts)]
    n_clusters = len(ordered) - (1 if uncategorised in ordered else 0)
    subtitle   = (
        " · ".join(hop_parts)
        + f" · {n_clusters} clusters"
        + (f" · {n_edges_drawn:,} edges" if n_edges_drawn else "")
    )

    fig.update_layout(
        title=dict(
            text=f"OpenFF Citation Landscape<br><sup>{subtitle}</sup>",
            font=dict(size=22, color="white"), x=0.5,
        ),
        width=1600, height=920,
        hovermode="closest",
        hoverlabel=dict(bgcolor="rgba(20,20,30,0.96)", font_size=13, font_color="white"),
        showlegend=True,
        legend=dict(
            title=dict(text="Topic cluster", font=dict(color="white", size=12)),
            x=1.01, y=0.99,
            font=dict(size=10, color="white"),
            bgcolor="rgba(0,0,0,0.45)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
            tracegroupgap=4,
            itemclick="toggle",           # single-click toggles one cluster (multi-select)
            itemdoubleclick="toggleothers",  # double-click to isolate a cluster
        ),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="white"),
        margin=dict(l=20, r=300, t=80, b=20),
    )

    # ── Generate base HTML ────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = fig.to_html(
        include_plotlyjs="cdn", full_html=True,
        config={"scrollZoom": True, "displayModeBar": True},
    )

    # ── Build overlay HTML + JS ───────────────────────────────────────────────
    today        = _date.today().isoformat()
    md           = metadata or {}
    n_total      = sum(hop_counts.values())
    pct_abstract = md.get("pct_abstract", "?")
    type_note  = (
        "" if md.get("has_type")
        else "<br><i style='color:#aaa'>Note: article types require a fresh --no-cache run.</i>"
    )

    # Hop toggle control box
    hop_checkboxes = ""
    symbols = {0: "■", 1: "●", 2: "○", 3: "○"}
    for hop in range(max_hop + 1):
        label      = HOP_LABELS.get(hop, f"Hop {hop}")
        sym        = symbols.get(hop, "●")
        cnt        = hop_counts.get(hop, 0)
        checked    = "checked" if hop < 2 else ""   # hop-2 starts unchecked
        hop_checkboxes += (
            f'<label style="display:block;margin:5px 0;cursor:pointer;font-size:12px;">'
            f'<input type="checkbox" id="toggle-hop-{hop}" {checked} '
            f'onchange="toggleHop({hop})">&nbsp;{sym} {label} ({cnt:,})'
            f'</label>\n'
        )

    control_box = f"""
<div id="hop-control-box" style="
  position:fixed; top:80px; left:16px; z-index:1000;
  background:rgba(10,12,18,0.92); border:1px solid rgba(255,255,255,0.15);
  border-radius:8px; padding:12px 16px; color:white;
  font-family:Arial,sans-serif; min-width:220px;
  box-shadow:0 2px 12px rgba(0,0,0,0.5);
">
  <div style="font-weight:bold;font-size:13px;margin-bottom:8px;color:#e0e0e0;">
    Show neighbourhoods
  </div>
  {hop_checkboxes}
  <hr style="border:none;border-top:1px solid rgba(255,255,255,0.15);margin:8px 0;">
  <button id="mode-toggle-btn" onclick="toggleDarkMode()" style="
    background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.2);
    color:white; cursor:pointer; font-size:11px; padding:5px 8px;
    border-radius:4px; width:100%; text-align:left; font-family:Arial,sans-serif;">
    ☀ Light mode
  </button>
</div>"""

    # Info box (collapsible)
    n_steps_str = md.get("n_steps", "?")
    info_box = f"""
<div id="info-box" style="
  position:fixed; bottom:16px; left:16px; z-index:1000;
  background:rgba(10,12,18,0.92); border:1px solid rgba(255,255,255,0.15);
  border-radius:8px; padding:12px 16px; color:white;
  font-family:Arial,sans-serif; max-width:500px; font-size:12px;
  box-shadow:0 2px 12px rgba(0,0,0,0.5);
">
  <div style="font-weight:bold;font-size:13px;margin-bottom:4px;color:#e0e0e0;display:flex;justify-content:space-between;align-items:center;">
    <span>About this map</span>
    <span id="info-toggle-btn" onclick="toggleInfo()"
      style="cursor:pointer;font-size:11px;color:#aaa;margin-left:12px;">&#x25BC; hide</span>
  </div>
  <div id="info-content" style="max-height:65vh; overflow-y:auto; padding-right:4px;">
    <p style="margin:6px 0;">
      {n_total:,} scientific papers arranged by semantic similarity
      (title, abstract &amp; keywords) and coloured by topic cluster.
    </p>
    <p style="margin:6px 0;">
      <b>Corpus</b> ({n_steps_str} citation hops from OpenFF)<br>
      {'<br>'.join(
          f'&nbsp;&nbsp;{symbols.get(h,"●")} Hop {h}: {hop_counts[h]:,} papers'
          for h in sorted(hop_counts)
      )}<br>
      &nbsp;&nbsp;Edges: {n_edges_drawn:,} shown (of {md.get("n_edges_total", len(edges)):,} total)
    </p>
    <p style="margin:6px 0;">
      <b>Method</b><br>
      &nbsp;&nbsp;Embeddings: all-MiniLM-L6-v2<br>
      &nbsp;&nbsp;Layout: UMAP (cosine, 2D)<br>
      &nbsp;&nbsp;Clusters: HDBSCAN &rarr; {n_clusters} topics<br>
      &nbsp;&nbsp;Labels: TF-IDF lift scoring
      {type_note}
    </p>
    <p style="margin:6px 0;">
      <b>Data source:</b> <a href="https://openalex.org" target="_blank"
        style="color:#7ec8e3">OpenAlex</a><br>
      <b>Last updated:</b> {today}
    </p>
    <p style="margin:6px 0;color:#aaa;">
      Hover to inspect &nbsp;|&nbsp; Click to open DOI<br>
      Scroll to zoom &nbsp;|&nbsp; Drag to pan
    </p>
    <div style="margin-top:8px;border-top:1px solid rgba(255,255,255,0.12);padding-top:6px;">
      <span id="details-toggle-btn" onclick="toggleDetails()"
        style="cursor:pointer;color:#7ec8e3;font-size:11px;">&#x25B6; More details</span>
      <div id="details-content" style="display:none;margin-top:6px;font-size:11px;color:#ccc;line-height:1.5;">
        <p style="margin:5px 0;"><b style="color:#e0e0e0;">Embeddings — all-MiniLM-L6-v2</b><br>
        A 22 M-parameter sentence transformer fine-tuned on over 1 billion sentence pairs for
        semantic textual similarity. It produces 384-dimensional unit vectors, striking a good
        balance between quality and speed for large corpora. Being open-source (Apache 2.0) and
        runnable locally avoids API costs and keeps the pipeline fully reproducible.<br>
        <b>Input:</b> title + abstract + OpenAlex keywords/concepts (repeated ×{KEYWORD_REPEAT_EMBED}
        for emphasis) + author names.<br>
        <span style="color:#f0ad4e;">⚠ Caveat:</span> {pct_abstract}% of papers have abstracts
        in OpenAlex; the remainder are embedded on title only (~20 words vs ~200+ for a full
        abstract), which produces noisier embeddings and may cause misclustering for those
        papers.</p>
        <p style="margin:5px 0;"><b style="color:#e0e0e0;">Layout — UMAP</b><br>
        UMAP preserves both local neighbourhood structure <i>and</i> global topology.
        The manifold is <b>fit on {md.get("n_fit_umap","?")} hop-0+1 papers</b> only
        (<code>fit_transform</code>), keeping the layout centred on OpenFF-relevant
        structure (<code>n_neighbors=15</code>, <code>min_dist=0.05</code>, cosine).
        Hop-2 papers are then <b>projected</b> into this space via UMAP's approximate
        <code>transform()</code> and shown as a toggleable layer (off by default).<br>
        <span style="color:#f0ad4e;">⚠ Caveat:</span> HDBSCAN clusters are computed
        in 2D UMAP space (not the original 384-dim space), so cluster boundaries are
        influenced by the projection.  Hop-2 positions are approximate.</p>
        <p style="margin:5px 0;"><b style="color:#e0e0e0;">Clustering — HDBSCAN</b><br>
        HDBSCAN requires no pre-specified cluster count and naturally labels low-density points
        as noise ("Uncategorised"), which avoids forcing unrelated papers into spurious clusters.
        It handles the variable-density structure common in citation landscapes and is
        deterministic via the excess-of-mass (EOM) leaf selection strategy.<br>
        <span style="color:#f0ad4e;">⚠ Caveat:</span> Clustering runs on the 2D UMAP projection
        (euclidean distance) rather than the original 384-dim embedding space (cosine). UMAP
        distorts long-range distances, so cluster boundaries are influenced by the projection.
        Clustering in high-dim space would be more rigorous but is ~10× slower at this scale.</p>
        <p style="margin:5px 0;"><b style="color:#e0e0e0;">Cluster labels — TF-IDF lift</b><br>
        Text is first processed with <b>Gensim Phrases</b> (bigram then trigram pass) to join
        statistically significant collocations into single tokens — e.g. "molecular dynamics",
        "neural network potentials", "free energy perturbation" — so they are treated as atomic
        units rather than split words.<br>
        Term lift = (centroid-weighted mean TF-IDF in cluster) / (smoothed mean TF-IDF in rest).
        Documents are weighted by proximity to the cluster centroid so the most representative
        papers drive the label. Keywords are repeated ×{KEYWORD_REPEAT} to outweigh free text.
        Author names, URL tokens, and generic scientific hedging terms ("proposed", "results", etc.)
        are excluded from the vocabulary. The top 3 terms by lift score are selected.</p>
        <p style="margin:5px 0;"><b style="color:#e0e0e0;">Data — OpenAlex</b><br>
        OpenAlex is a free, open scholarly graph of ~250 M works with a permissive API.
        Unlike Web of Science or Scopus it requires no subscription and supports programmatic
        citation-graph traversal, making it well suited for fully automated, reproducible
        pipeline runs.<br>
        <span style="color:#f0ad4e;">⚠ Caveats:</span> (1) Reference extraction coverage is
        estimated at ~80–85%; some citation edges are missing from the graph.
        (2) Hop-2 papers cite papers that cite OpenFF — their connection to OpenFF may be
        indirect and incidental; treat them as context rather than direct citations.</p>
        <p style="margin:5px 0;"><b style="color:#e0e0e0;">Assumptions &amp; known limitations</b>
        <ul style="margin:4px 0 4px 12px;padding:0;color:#ccc;">
          <li><b>Edge direction:</b> arrows are not shown; edges run from the citing paper toward the cited paper.</li>
          <li><b>Citation intent:</b> all citations are counted equally — a paper mentioning OpenFF in passing receives the same hop-1 status as one that extensively uses it.</li>
          <li><b>Recency lag:</b> <code>cited_by_count</code> (used for node size) lags by weeks to months for newly published papers.</li>
          <li><b>Duplicate works:</b> preprints and their published journal versions, or parallel Zenodo versions of the same software release, may appear as separate nodes with nearly identical embeddings.</li>
          <li><b>Zenodo version records:</b> each versioned Zenodo DOI is a distinct hop-0 node; "openff-2.1.0" and "openff-2.2.0" are separate points with near-identical titles, creating artificial density clumps in hop-0.</li>
          <li><b>Edge budget:</b> only {max_edges:,} of {md.get("n_edges_total", len(edges)):,} total citation edges are shown; low-hop edges are kept first.</li>
          <li><b>OpenAlex coverage:</b> papers not yet indexed, or indexed without a DOI match, are absent from the graph entirely.</li>
        </ul></p>
      </div>
    </div>
  </div>
</div>"""

    # JavaScript
    hop_traces_js       = json.dumps(hop_traces)
    hop_point_traces_js = json.dumps(hop_point_traces)
    hop_legend_js       = json.dumps(hop_legend_indices)
    type_legend_js      = json.dumps(type_legend_indices)
    edge_trace_js       = json.dumps(edge_trace_map)
    # Per-hop dark-mode outline colors (mirrors HOP_STYLE border_color)
    hop_dark_outlines   = {str(h): HOP_STYLE[min(h, max_style)][3]
                           for h in range(max_hop + 1)}
    hop_dark_outlines_js = json.dumps(hop_dark_outlines)
    js_code = f"""
<script>
var HOP_TRACES        = {hop_traces_js};
var HOP_POINT_TRACES  = {hop_point_traces_js};
var HOP_LEGEND_TRACES = {hop_legend_js};
var TYPE_LEGEND_TRACES= {type_legend_js};
var EDGE_TRACES       = {edge_trace_js};
var HOP_DARK_OUTLINES = {hop_dark_outlines_js};
var _infoOpen = true;
var _darkMode = true;
var THEME = {{
  dark: {{
    plotBg: '#0f1117', paperBg: '#0f1117', fontColor: 'white',
    boxBg: 'rgba(10,12,18,0.92)', boxBorder: 'rgba(255,255,255,0.15)',
    btnBg: 'rgba(255,255,255,0.1)', btnBorder: 'rgba(255,255,255,0.2)', btnColor: 'white',
    legendBg: 'rgba(0,0,0,0.45)', legendBorder: 'rgba(255,255,255,0.15)',
    btnLabel: '☀ Light mode',
    legendMarkerColor: 'white',
    edgeColor: 'rgba(200,200,200,0.15)',
  }},
  light: {{
    plotBg: '#f8f9fa', paperBg: '#f8f9fa', fontColor: '#111',
    boxBg: 'rgba(245,245,245,0.96)', boxBorder: 'rgba(0,0,0,0.18)',
    btnBg: 'rgba(0,0,0,0.07)', btnBorder: 'rgba(0,0,0,0.18)', btnColor: '#111',
    legendBg: 'rgba(255,255,255,0.92)', legendBorder: 'rgba(0,0,0,0.18)',
    btnLabel: '🌙 Dark mode',
    legendMarkerColor: '#333',
    edgeColor: 'rgba(60,60,60,0.2)',
    // per-hop light-mode marker outlines (darker for higher contrast on white)
    hopOutline: {{'0':'rgba(20,20,20,0.9)', '1':'rgba(20,20,20,0.5)', '2':'rgba(20,20,20,0.35)', '3':'rgba(20,20,20,0.25)'}},
  }},
}};

function toggleDarkMode() {{
  _darkMode = !_darkMode;
  var t = _darkMode ? THEME.dark : THEME.light;
  var gd = document.querySelector('.js-plotly-plot');
  if (gd) {{
    Plotly.relayout(gd, {{
      'paper_bgcolor': t.paperBg,
      'plot_bgcolor':  t.plotBg,
      'font.color':    t.fontColor,
      'title.font.color': t.fontColor,
      'legend.bgcolor':      t.legendBg,
      'legend.bordercolor':  t.legendBorder,
      'legend.font.color':   t.fontColor,
    }});

    // Restyle marker outlines per hop (real point traces only)
    Object.keys(HOP_POINT_TRACES).forEach(function(hop) {{
      var indices = HOP_POINT_TRACES[hop];
      if (!indices || !indices.length) return;
      var outline = _darkMode
        ? HOP_DARK_OUTLINES[hop]
        : (t.hopOutline[hop] || t.hopOutline['1']);
      Plotly.restyle(gd, {{'marker.line.color': outline}}, indices);
    }});

    // Restyle hop legend dummy markers
    if (HOP_LEGEND_TRACES.length) {{
      Plotly.restyle(gd,
        {{'marker.color': t.legendMarkerColor, 'marker.line.color': t.legendMarkerColor}},
        HOP_LEGEND_TRACES);
    }}

    // Restyle article-type legend dummy markers
    if (TYPE_LEGEND_TRACES.length) {{
      Plotly.restyle(gd, {{'marker.color': t.legendMarkerColor}}, TYPE_LEGEND_TRACES);
    }}

    // Restyle edge traces
    var edgeIndices = Object.values(EDGE_TRACES);
    if (edgeIndices.length) {{
      Plotly.restyle(gd, {{'line.color': t.edgeColor}}, edgeIndices);
    }}
  }}
  document.body.style.background = t.plotBg;
  [document.getElementById('hop-control-box'),
   document.getElementById('info-box')].forEach(function(el) {{
    if (!el) return;
    el.style.background  = t.boxBg;
    el.style.borderColor = t.boxBorder;
    el.style.color       = t.fontColor;
  }});
  var btn = document.getElementById('mode-toggle-btn');
  if (btn) {{
    btn.style.background  = t.btnBg;
    btn.style.borderColor = t.btnBorder;
    btn.style.color       = t.btnColor;
    btn.textContent       = t.btnLabel;
  }}
}}

function toggleHop(hop) {{
  var gd = document.querySelector('.js-plotly-plot');
  if (!gd) return;
  var checked = document.getElementById('toggle-hop-' + hop).checked;
  var indices = HOP_TRACES[hop] || [];
  if (indices.length > 0) {{
    Plotly.restyle(gd, {{visible: checked}}, indices);
  }}
  updateEdgeTraces();
}}

function updateEdgeTraces() {{
  var gd = document.querySelector('.js-plotly-plot');
  if (!gd) return;
  var vis = {{}};
  for (var h in HOP_TRACES) {{
    var cb = document.getElementById('toggle-hop-' + h);
    vis[parseInt(h)] = cb ? cb.checked : true;
  }}
  for (var key in EDGE_TRACES) {{
    var parts = key.split('-');
    var lo = parseInt(parts[0]), hi = parseInt(parts[1]);
    var show = (vis[lo] !== false) && (vis[hi] !== false);
    Plotly.restyle(gd, {{visible: show}}, [EDGE_TRACES[key]]);
  }}
}}

function toggleInfo() {{
  _infoOpen = !_infoOpen;
  document.getElementById('info-content').style.display = _infoOpen ? 'block' : 'none';
  document.getElementById('info-toggle-btn').innerHTML = _infoOpen
    ? '&#x25BC; hide' : '&#x25B6; show';
}}

var _detailsOpen = false;
function toggleDetails() {{
  _detailsOpen = !_detailsOpen;
  document.getElementById('details-content').style.display = _detailsOpen ? 'block' : 'none';
  document.getElementById('details-toggle-btn').innerHTML = _detailsOpen
    ? '&#x25BC; Less details' : '&#x25B6; More details';
}}

(function attach() {{
  var divs = document.querySelectorAll('.js-plotly-plot');
  if (!divs.length) {{ setTimeout(attach, 400); return; }}
  divs.forEach(function(div) {{
    div.on('plotly_click', function(data) {{
      var pt = data.points[0];
      if (pt && pt.customdata) window.open(pt.customdata, '_blank');
    }});
  }});
}})();
</script>"""

    inject = control_box + info_box + js_code
    html = html.replace("</body>", inject + "\n</body>")
    output_path.write_text(html, encoding="utf-8")
    print(f"\n✓ Saved → {output_path}")
    print(f"  Open:  open '{output_path}'")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="OpenFF Citation Analysis Pipeline")
    ap.add_argument("--publications", default=str(DEFAULT_PUBLICATIONS))
    ap.add_argument("--zenodo",       default=str(DEFAULT_ZENODO))
    ap.add_argument(
        "--n-steps", type=int, default=2,
        help="Number of citation hops beyond OpenFF papers "
             "(0=OpenFF only, 1=direct citers, 2=indirect citers, …)",
    )
    ap.add_argument("--output",    default=str(OUTPUTS_DIR / "citations.html"))
    ap.add_argument("--email",     default="", help="Email for OpenAlex polite pool")
    ap.add_argument("--no-cache",  action="store_true", help="Ignore / overwrite cached data")
    ap.add_argument("--min-cluster-size", type=int, default=10,
                    help="HDBSCAN min_cluster_size; larger = fewer coarser clusters")
    ap.add_argument("--min-samples", type=int, default=2,
                    help="HDBSCAN min_samples; controls noise sensitivity")
    ap.add_argument(
        "--max-edges", type=int, default=10_000,
        help="Maximum edges drawn (randomly subsampled if exceeded)",
    )
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Sources
    sources = load_openff_sources(Path(args.publications), Path(args.zenodo))
    print(f"Loaded {len(sources)} OpenFF source DOIs")

    # 2. Corpus
    # Always provide cache_path so fresh data is saved; force_refresh skips loading.
    corpus_cache = DATA_DIR / f"corpus_{CACHE_VERSION}_n{args.n_steps}.json"
    papers, cites_openff, roles, edges, openff_works = collect_corpus(
        sources, n_steps=args.n_steps, email=args.email,
        cache_path=corpus_cache, force_refresh=args.no_cache,
    )

    # 3. DataFrame — all hops
    df = build_dataframe(papers, cites_openff, roles, sources)
    df.to_csv(DATA_DIR / "papers.csv", index=False)
    hop2_count = int((df["hop"] > 1).sum())
    print(f"\nDataFrame: {len(df)} papers  "
          f"(hop-0: {int((df['hop']==0).sum())}, "
          f"hop-1: {int((df['hop']==1).sum())}, "
          f"hop-2: {hop2_count})")

    if len(df) < 10:
        print("Too few papers to proceed. Exiting.")
        return

    # Boolean mask: which rows are used to FIT the UMAP manifold (hop-0+1 only)
    fit_mask = (df["hop"] <= 1).values

    # 4. Embed — all papers; always cache; force_refresh skips loading stale file
    emb_cache = DATA_DIR / f"embeddings_{CACHE_VERSION}.npy"
    embs = embed(df, cache_path=emb_cache, force_refresh=args.no_cache)

    # 5. UMAP — fit on hop-0+1, transform hop-2 (approximate) into that layout
    umap_cache = DATA_DIR / f"umap_coords_{CACHE_VERSION}.npy"
    coords = run_umap(
        embs, fit_mask=fit_mask,
        cache_path=umap_cache, force_refresh=args.no_cache,
    )

    # 6. Cluster — only on hop-0+1 coords; hop-2 papers all get label -1
    labels_fit    = cluster_hdbscan(coords[fit_mask],
                                    min_cluster_size=args.min_cluster_size,
                                    min_samples=args.min_samples)
    cluster_labels = np.full(len(df), -1, dtype=int)
    cluster_labels[fit_mask] = labels_fit

    # 7. Label clusters (using hop-0+1 subset only)
    df_fit = df[fit_mask].reset_index(drop=True)
    cluster_names = label_clusters(df_fit, labels_fit, coords[fit_mask])
    cluster_names[-1] = "Uncategorised"

    # 8. Visualise
    hop_counts       = df.groupby("hop").size().to_dict()
    has_type         = df["article_type"].ne("").any()
    n_with_abstract  = int((df["abstract"] != "").sum())
    pct_abstract     = round(100 * n_with_abstract / max(len(df), 1))
    n_fit_umap       = int(fit_mask.sum())
    metadata   = {
        "n_steps":        args.n_steps,
        "hop_counts":     hop_counts,
        "n_edges_total":  len(edges),
        "has_type":       bool(has_type),
        "pct_abstract":   pct_abstract,
        "n_fit_umap":     n_fit_umap,
    }
    create_viz(
        df, coords, cluster_labels, cluster_names,
        edges, Path(args.output),
        max_edges=args.max_edges,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
