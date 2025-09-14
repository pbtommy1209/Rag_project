# Enhanced RAG Framework (PDF → FAISS → Ollama Answering)

This project provides a comprehensive RAG (Retrieval-Augmented Generation) framework that ingests PDFs into FAISS vector indexes using **Ollama embeddings**, retrieves relevant passages, and generates answers with citations using **Ollama chat models**.

## 🚀 Quick Start

### Complete Workflow (Recommended)
```bash
# Run the complete workflow from ingestion to optimization
python run_complete_workflow.py --pdf "F1_33.pdf" --outdir "./faiss_indexes"
```

## 📁 Project Structure

### Core Files
- `config.py` — enhanced central configuration
- `ollama_client.py` — HTTP client for Ollama embeddings & chat
- `ingest.py` — original PDF ingestion (basic)
- `retrieve.py` — original retrieval (basic)
- `answer.py` — RAG answer generation with citations
- `eval.py` — original evaluation (basic)
- `run_demo.py` — interactive CLI demo

### Enhanced Files
- `enhanced_ingest.py` — **NEW**: Multiple chunking strategies (fixed, semantic, adaptive, hierarchical)
- `enhanced_retrieve.py` — **NEW**: Hybrid search (semantic + keyword) with reranking
- `enhanced_eval.py` — **NEW**: Comprehensive evaluation with multiple metrics
- `optimize_params.py` — **NEW**: Automated parameter optimization
- `run_complete_workflow.py` — **NEW**: Complete end-to-end workflow

## 🛠️ Prerequisites

- **Python 3.10+**
- **Ollama** running locally (default `http://localhost:11434`). Install: https://ollama.com

### Install Ollama Models
```bash
ollama pull mxbai-embed-large      # embedding model (recommended)
ollama pull llama3.2:3b            # chat model (or llama3.1:8b for better quality)
```

## 📦 Setup

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for enhanced chunking)
python -c "import nltk; nltk.download('punkt')"
```

## 🔧 Usage

### 1. Enhanced Ingestion (Multiple Chunking Strategies)
```bash
python enhanced_ingest.py \
    --pdf "F1_33.pdf" \
    --outdir "./faiss_indexes" \
    --strategy all \
    --sizes "200,300,400,500,600,800,1000" \
    --overlaps "25,50,75,100,125,150,200" \
    --embed-model "mxbai-embed-large"
```

### 2. Enhanced Retrieval (Hybrid Search)
```bash
python enhanced_retrieve.py \
    --index "./faiss_indexes/adaptive_size500_overlap100" \
    --query "Who is Max Verstappen?" \
    --method hybrid \
    --alpha 0.7 \
    --rerank \
    --k 5
```

### 3. Comprehensive Evaluation
```bash
python enhanced_eval.py \
    --pdf "F1_33.pdf" \
    --indexes_root "./faiss_indexes" \
    --embed-model "mxbai-embed-large" \
    --chat-model "llama3.2:3b" \
    --k 5 \
    --output "evaluation_results.json"
```

### 4. Parameter Optimization
```bash
python optimize_params.py \
    --indexes_root "./faiss_indexes" \
    --embed-model "mxbai-embed-large" \
    --chat-model "llama3.2:3b" \
    --max-combinations 50 \
    --output "optimization_results.json"
```

### 5. Interactive Demo
```bash
python run_demo.py \
    --index "./faiss_indexes/adaptive_size500_overlap100" \
    --embed-model "mxbai-embed-large" \
    --chat-model "llama3.2:3b" \
    --k 5
```

## 🎯 Key Features

### Enhanced Chunking Strategies
- **Fixed-size**: Original sliding window approach
- **Semantic**: Sentence-based chunking with overlap
- **Adaptive**: Respects sentence boundaries
- **Hierarchical**: Multiple granularities in one index

### Advanced Retrieval Methods
- **Semantic**: Vector similarity search (original)
- **Keyword**: TF-IDF based search
- **Hybrid**: Combines semantic + keyword with configurable weights
- **Reranking**: Post-retrieval relevance scoring

### Comprehensive Evaluation
- **Retrieval Metrics**: Hit@k, MRR, NDCG
- **Answer Quality**: Exact Match, BLEU, ROUGE
- **Question Types**: Factual, Reasoning, Comparative, Numerical
- **Performance Analysis**: By question type and parameter

### Automated Optimization
- **Grid Search**: Systematic parameter exploration
- **Performance Analysis**: Parameter contribution analysis
- **Best Configuration**: Automatic optimal parameter discovery

## 📊 Evaluation Metrics

- **Hit@k**: Percentage of questions where correct answer is in top-k retrieved passages
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of first relevant passage
- **NDCG (Normalized Discounted Cumulative Gain)**: Ranking quality metric
- **Exact Match**: Percentage of questions answered correctly
- **Answer Quality**: BLEU, ROUGE scores for answer generation

## 🔍 Chunking Strategy Comparison

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| Fixed-size | Simple, fast | May break sentences | General purpose |
| Semantic | Preserves meaning | Variable chunk sizes | Complex documents |
| Adaptive | Balanced approach | More complex | Most use cases |
| Hierarchical | Multiple granularities | Larger index size | Complex queries |

## 🎛️ Configuration

Edit `config.py` to customize:
- Model choices (embedding, chat)
- Chunking parameters
- Retrieval methods
- Evaluation settings
- Optimization parameters

## 📈 Performance Tips

1. **Chunk Size**: 300-800 characters work well for most documents
2. **Overlap**: 10-20% of chunk size provides good context
3. **Top-K**: 5-10 retrieved passages balance relevance and noise
4. **Hybrid Alpha**: 0.6-0.8 for semantic, 0.2-0.4 for keyword weight
5. **Model Choice**: Larger models (8B+) provide better answer quality

## 🐛 Troubleshooting

- **Ollama Connection**: Ensure Ollama is running on `http://localhost:11434`
- **Model Not Found**: Pull required models with `ollama pull <model-name>`
- **Memory Issues**: Reduce batch size in ingestion or use smaller models
- **Slow Performance**: Use smaller embedding models or reduce chunk sizes

## 📝 Notes

- Uses cosine similarity (L2-normalized vectors, FAISS `IndexFlatIP`)
- RAG prompts include citations as `[p1]`, `[p2]`, etc.
- Supports multiple PDF formats through PyPDF
- Configurable for different document types and use cases
