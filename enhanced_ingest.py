#!/usr/bin/env python3
# enhanced_ingest.py
# Improved PDF ingestion with resilient NLTK handling and an 'adaptive' chunking strategy.
# Usage example:
#   python enhanced_ingest.py --pdf F1_33.pdf --outdir ./simple_index --strategy adaptive \
#       --sizes "500" --overlaps "100" --embed-model "mxbai-embed-large" --batch-size 3

import os, argparse, json, re, sys, math
from typing import List, Dict
import numpy as np
from pypdf import PdfReader
import faiss

# Try importing tqdm; if missing, provide no-op fallback
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

# --------- NLTK resilient import + fallbacks ----------
_HAS_NLTK = False
try:
    import nltk  # type: ignore
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize  # type: ignore
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            pass
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False

if _HAS_NLTK:
    def sent_tokenize(text: str) -> List[str]:
        return _nltk_sent_tokenize(text)
else:
    _sent_split_re = re.compile(r'(?<=[.!?])\s+')
    def sent_tokenize(text: str) -> List[str]:
        sents = [s.strip() for s in _sent_split_re.split(text) if s and s.strip()]
        if not sents:
            sents = [line.strip() for line in text.splitlines() if line.strip()]
        return sents

# --------- util helpers ----------
def print_env_diagnostics():
    print("=== Environment diagnostics ===", file=sys.stderr)
    print(f"Python executable: {sys.executable}", file=sys.stderr)
    print(f"sys.path[0:3]: {sys.path[:3]}", file=sys.stderr)
    print(f"Has NLTK: {_HAS_NLTK}", file=sys.stderr)
    print("===============================", file=sys.stderr)

def read_pdf_enhanced(path: str) -> str:
    r = PdfReader(path)
    pages = []
    for p in r.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        t = re.sub(r"\s+\n", "\n", t)
        t = re.sub(r"[ \t]+", " ", t)
        pages.append(t.strip())
    return "\n\n".join(pages).strip()

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def build_faiss(index_dir: str, embeddings: np.ndarray, meta: List[dict]):
    os.makedirs(index_dir, exist_ok=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# --------- chunkers ----------
def chunk_text_by_chars(text: str, size: int = 500, overlap: int = 100) -> List[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + size, n)
        chunks.append(text[i:j].strip())
        if j == n:
            break
        i = j - overlap
    return [c for c in chunks if c]

def chunk_text_by_sentences(text: str, approx_size_chars: int = 500, overlap_chars: int = 100) -> List[str]:
    sents = sent_tokenize(text)
    chunks = []
    cur = []
    cur_len = 0
    for s in sents:
        s_len = len(s)
        if cur and cur_len + s_len > approx_size_chars:
            chunks.append(" ".join(cur).strip())
            cur = [s]
            cur_len = s_len
        else:
            cur.append(s)
            cur_len += s_len + 1
    if cur:
        chunks.append(" ".join(cur).strip())
    # add character overlap to maintain boundary context
    if overlap_chars > 0 and len(chunks) > 1:
        out = []
        for i, c in enumerate(chunks):
            if i == 0:
                out.append(c)
                continue
            prev = out[-1]
            overlap_text = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
            combined = (overlap_text + " " + c).strip()
            out.append(combined)
        chunks = out
    return [c for c in chunks if c]

def chunk_text_adaptive(text: str, size: int = 500, overlap: int = 100) -> List[str]:
    """
    Adaptive: prefer sentence-based chunks, but:
      - if sentence-based chunk > 1.5*size, split that chunk into char-windows,
      - if sentence-based produced zero chunks (weird PDF), fallback to char-windows.
    This balances semantic coherence and strict size control.
    """
    sent_chunks = chunk_text_by_sentences(text, approx_size_chars=size, overlap_chars=overlap)
    if not sent_chunks:
        return chunk_text_by_chars(text, size=size, overlap=overlap)

    final = []
    max_allowed = int(size * 1.5)
    for sc in sent_chunks:
        if len(sc) > max_allowed:
            # split large sentence-chunk into char windows with same overlap
            sub = chunk_text_by_chars(sc, size=size, overlap=overlap)
            final.extend(sub)
        else:
            final.append(sc)
    # ensure final has no empties
    return [c for c in final if c]

# --------- dummy embedder fallback (for debugging) ----------
def dummy_embed(texts: List[str]) -> np.ndarray:
    vecs = []
    for t in texts:
        h = abs(hash(t)) % (10**8)
        v = np.array([(h >> (i*3)) & 255 for i in range(64)], dtype=np.float32)
        vecs.append(v)
    X = np.vstack(vecs).astype(np.float32)
    return l2_normalize(X)

# --------- embedding helper w/ batching ----------
def embed_with_client(client, model_name: str, texts: List[str], batch_size: int = 8):
    """
    Use client's embed_batch in batches. Expects client.embed_batch(list_of_texts) OR
    client.embed_single(text) fallback handled.
    """
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Many Ollama client implementations accept list or single; try both
        try:
            vecs = client.embed_batch(model_name, batch)
        except TypeError:
            # maybe embed_batch expects single text; call repeatedly
            vecs = []
            for t in batch:
                vecs.append(client.embed_batch(model_name, [t])[0])
        all_vecs.extend(vecs)
    return np.array(all_vecs, dtype=np.float32)

# --------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--strategy", choices=["char", "sentence", "adaptive"], default="adaptive")
    ap.add_argument("--sizes", type=str, default="500")
    ap.add_argument("--overlaps", type=str, default="100")
    ap.add_argument("--embed-model", type=str, default="mxbai-embed-large:latest")
    ap.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--use-dummy-embed", action="store_true")
    args = ap.parse_args()

    print_env_diagnostics()
    text = read_pdf_enhanced(args.pdf)
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    overlaps = [int(x) for x in args.overlaps.split(",") if x.strip()]

    for size in sizes:
        for ov in overlaps:
            if ov >= size:
                print(f"Skipping size={size} overlap={ov} because overlap >= size", file=sys.stderr)
                continue

            if args.strategy == "char":
                chunks = chunk_text_by_chars(text, size=size, overlap=ov)
            elif args.strategy == "sentence":
                chunks = chunk_text_by_sentences(text, approx_size_chars=size, overlap_chars=ov)
            else:  # adaptive
                chunks = chunk_text_adaptive(text, size=size, overlap=ov)

            print(f"Generated {len(chunks)} chunks for size={size} overlap={ov} (strategy={args.strategy})")

            if not chunks:
                print("No chunks created; skipping this config.", file=sys.stderr)
                continue

            # embed
            if args.use_dummy_embed:
                X = dummy_embed(chunks)
            else:
                try:
                    from ollama_client import OllamaClient  # your project's client
                    client = OllamaClient(base_url=args.ollama_url)
                    X = embed_with_client(client, args.embed_model, chunks, batch_size=args.batch_size)
                    X = l2_normalize(X)
                except Exception as e:
                    print(f"Embedding failed with error: {e}", file=sys.stderr)
                    print("Falling back to dummy embeddings for debugging.", file=sys.stderr)
                    X = dummy_embed(chunks)

            meta = [{"id": i, "chunk": chunks[i]} for i in range(len(chunks))]
            subdir = os.path.join(args.outdir, f"size{size}_overlap{ov}")
            build_faiss(subdir, X, meta)
            print(f"Built index at {subdir} with dim={X.shape[1]} and chunks={len(chunks)}")

if __name__ == "__main__":
    import argparse
    main()
