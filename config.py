# config.py — enhanced central configuration
DEFAULT_PDF_PATH = "/Users/anothpbt/Documents/rag_project/F1_33.pdf"

# Vector / retrieval
TOP_K = 5
EMBED_MODEL = "mxbai-embed-large:latest"   # try: "nomic-embed-text", "bge-m3"
CHAT_MODEL  = "llama3.2:3b"         # try: "llama3.1:8b", "qwen2.5:7b"

# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"  # change if remote
TIMEOUT = 120

# Chunking defaults
CHUNK_SIZES   = [300, 500, 800, 1000]
CHUNK_OVERLAPS = [50, 100, 150, 200]

# Enhanced chunking strategies
CHUNKING_STRATEGIES = ["fixed", "semantic", "adaptive", "hierarchical"]

# Retrieval methods
RETRIEVAL_METHODS = ["semantic", "keyword", "hybrid"]
HYBRID_ALPHA = 0.7  # Weight for semantic vs keyword in hybrid search

# Evaluation settings
EVALUATION_METRICS = ["hit@1", "hit@3", "hit@5", "hit@10", "mrr", "ndcg", "exact_match"]
QUESTION_TYPES = ["factual", "reasoning", "comparative", "numerical"]

# Optimization settings
MAX_OPTIMIZATION_COMBINATIONS = 50
OPTIMIZATION_METRIC = "exact_match"  # Primary metric for optimization

# Advanced features
ENABLE_RERANKING = False
ENABLE_CACHING = True
CACHE_SIZE = 1000
