# enhanced_retrieve.py — advanced retrieval with hybrid search and reranking
import os, json, argparse
import numpy as np
import faiss
from typing import List, Tuple, Dict
import re
from collections import Counter
import math

from ollama_client import OllamaClient

def load_index(index_dir: str):
    """Load FAISS index and metadata"""
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    # Handle both old format (list) and new format (dict with chunks key)
    if isinstance(meta, list):
        # Old format: convert to new format
        meta = {"chunks": meta}
    
    return index, meta

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """L2 normalization for embeddings"""
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text using simple frequency analysis"""
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    
    # Tokenize and clean
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    words = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Count frequencies
    word_counts = Counter(words)
    return [word for word, count in word_counts.most_common(max_keywords)]

def calculate_tfidf_score(query_keywords: List[str], chunk_text: str, all_chunks: List[str]) -> float:
    """Calculate TF-IDF score for a chunk"""
    chunk_words = re.findall(r'\b[a-zA-Z]+\b', chunk_text.lower())
    chunk_word_counts = Counter(chunk_words)
    
    # Safety check for empty chunks
    if not chunk_words or not all_chunks:
        return 0.0
    
    # Calculate TF-IDF for each query keyword
    tfidf_score = 0.0
    for keyword in query_keywords:
        # Term frequency in current chunk
        tf = chunk_word_counts.get(keyword, 0) / len(chunk_words)
        
        # Document frequency (how many chunks contain this keyword)
        df = sum(1 for chunk in all_chunks if keyword in chunk.lower())
        
        # Inverse document frequency
        idf = math.log(len(all_chunks) / (df + 1)) if df > 0 else 0
        
        tfidf_score += tf * idf
    
    return tfidf_score

def semantic_retrieve(index_dir: str, query: str, k: int, embed_model: str, ollama_url: str) -> List[Dict]:
    """Original semantic retrieval"""
    client = OllamaClient(ollama_url)
    qv = np.array(client.embed_batch(embed_model, [query])[0], dtype=np.float32).reshape(1, -1)
    qv = l2_normalize(qv)
    index, meta = load_index(index_dir)
    scores, I = index.search(qv, k=k)
    idxs = I[0].tolist()
    hits = [{"rank": r+1, "score": float(scores[0][r]), "id": i, "chunk": meta['chunks'][i]["chunk"]} 
            for r, i in enumerate(idxs)]
    return hits

def keyword_retrieve(index_dir: str, query: str, k: int) -> List[Dict]:
    """Keyword-based retrieval using TF-IDF"""
    index, meta = load_index(index_dir)
    chunks = [chunk['chunk'] for chunk in meta['chunks']]
    
    # Extract keywords from query
    query_keywords = extract_keywords(query)
    
    # Calculate TF-IDF scores
    scores = []
    for i, chunk in enumerate(chunks):
        tfidf_score = calculate_tfidf_score(query_keywords, chunk, chunks)
        scores.append((i, tfidf_score))
    
    # Sort by score and return top-k
    scores.sort(key=lambda x: x[1], reverse=True)
    hits = [{"rank": r+1, "score": float(score), "id": idx, "chunk": chunks[idx]} 
            for r, (idx, score) in enumerate(scores[:k])]
    return hits

def hybrid_retrieve(index_dir: str, query: str, k: int, embed_model: str, ollama_url: str, 
                   alpha: float = 0.7) -> List[Dict]:
    """Hybrid retrieval combining semantic and keyword search"""
    # Get semantic results
    semantic_hits = semantic_retrieve(index_dir, query, k*2, embed_model, ollama_url)
    
    # Get keyword results
    keyword_hits = keyword_retrieve(index_dir, query, k*2)
    
    # Create score maps
    semantic_scores = {hit['id']: hit['score'] for hit in semantic_hits}
    keyword_scores = {hit['id']: hit['score'] for hit in keyword_hits}
    
    # Normalize scores
    if semantic_hits:
        max_semantic = max(semantic_scores.values())
        semantic_scores = {k: v/max_semantic for k, v in semantic_scores.items()}
    
    if keyword_hits:
        max_keyword = max(keyword_scores.values())
        keyword_scores = {k: v/max_keyword for k, v in keyword_scores.items()}
    
    # Combine scores
    combined_scores = {}
    all_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
    
    for chunk_id in all_ids:
        semantic_score = semantic_scores.get(chunk_id, 0)
        keyword_score = keyword_scores.get(chunk_id, 0)
        combined_score = alpha * semantic_score + (1 - alpha) * keyword_score
        combined_scores[chunk_id] = combined_score
    
    # Sort by combined score
    sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get chunk texts
    index, meta = load_index(index_dir)
    chunks = {i: chunk['chunk'] for i, chunk in enumerate(meta['chunks'])}
    
    hits = [{"rank": r+1, "score": float(score), "id": chunk_id, "chunk": chunks[chunk_id],
             "semantic_score": semantic_scores.get(chunk_id, 0),
             "keyword_score": keyword_scores.get(chunk_id, 0)}
            for r, (chunk_id, score) in enumerate(sorted_scores[:k])]
    
    return hits

def rerank_results(hits: List[Dict], query: str, embed_model: str, ollama_url: str) -> List[Dict]:
    """Rerank results using cross-encoder style scoring"""
    if len(hits) <= 1:
        return hits
    
    client = OllamaClient(ollama_url)
    
    # Create reranking prompts
    rerank_prompts = []
    for hit in hits:
        prompt = f"Query: {query}\n\nPassage: {hit['chunk']}\n\nRate the relevance of this passage to the query on a scale of 1-10:"
        rerank_prompts.append(prompt)
    
    # Get relevance scores from LLM
    relevance_scores = []
    for prompt in rerank_prompts:
        try:
            response = client.chat(embed_model.replace('embed', ''), 
                                 [{"role": "user", "content": prompt}], 
                                 temperature=0.1)
            # Extract numeric score from response
            score_match = re.search(r'(\d+)', response)
            score = float(score_match.group(1)) if score_match else 5.0
            relevance_scores.append(min(max(score, 1), 10))  # Clamp to 1-10
        except:
            relevance_scores.append(5.0)  # Default score
    
    # Combine original scores with relevance scores
    for i, hit in enumerate(hits):
        hit['rerank_score'] = relevance_scores[i]
        hit['combined_score'] = 0.6 * hit['score'] + 0.4 * (relevance_scores[i] / 10)
    
    # Sort by combined score
    hits.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Update ranks
    for i, hit in enumerate(hits):
        hit['rank'] = i + 1
    
    return hits

def retrieve_enhanced(index_dir: str, query: str, k: int, embed_model: str, ollama_url: str,
                     method: str = "semantic", alpha: float = 0.7, rerank: bool = False) -> List[Dict]:
    """Enhanced retrieval with multiple methods"""
    
    if method == "semantic":
        hits = semantic_retrieve(index_dir, query, k, embed_model, ollama_url)
    elif method == "keyword":
        hits = keyword_retrieve(index_dir, query, k)
    elif method == "hybrid":
        hits = hybrid_retrieve(index_dir, query, k, embed_model, ollama_url, alpha)
    else:
        raise ValueError(f"Unknown retrieval method: {method}")
    
    if rerank and len(hits) > 1:
        hits = rerank_results(hits, query, embed_model, ollama_url)
    
    return hits

def main():
    ap = argparse.ArgumentParser(description="Enhanced retrieval with multiple methods")
    ap.add_argument("--index", required=True, help="FAISS index directory")
    ap.add_argument("--query", required=True, help="Query string")
    ap.add_argument("--k", type=int, default=5, help="Number of results to return")
    ap.add_argument("--method", type=str, default="semantic", 
                   choices=["semantic", "keyword", "hybrid"], help="Retrieval method")
    ap.add_argument("--alpha", type=float, default=0.7, 
                   help="Weight for semantic vs keyword in hybrid search")
    ap.add_argument("--rerank", action="store_true", help="Enable reranking")
    ap.add_argument("--embed-model", type=str, default="mxbai-embed-large")
    ap.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    args = ap.parse_args()

    hits = retrieve_enhanced(args.index, args.query, args.k, args.embed_model, args.ollama_url,
                           args.method, args.alpha, args.rerank)
    
    print(f"\n🔍 Results for: '{args.query}'")
    print(f"Method: {args.method.upper()}" + (" + RERANK" if args.rerank else ""))
    print("-" * 80)
    
    for hit in hits:
        score_info = f"score={hit['score']:.4f}"
        if 'rerank_score' in hit:
            score_info += f" | rerank={hit['rerank_score']:.1f} | combined={hit['combined_score']:.4f}"
        if 'semantic_score' in hit and 'keyword_score' in hit:
            score_info += f" | sem={hit['semantic_score']:.3f} | kw={hit['keyword_score']:.3f}"
        
        print(f"[{hit['rank']}] {score_info}")
        print(f"     {hit['chunk'][:200].replace(chr(10), ' ')}{'...' if len(hit['chunk']) > 200 else ''}")
        print()

if __name__ == "__main__":
    main()
