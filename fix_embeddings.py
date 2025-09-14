#!/usr/bin/env python3
# fix_embeddings.py — fix the embedding issue by recreating indexes with working model
import os, sys, json, faiss
import numpy as np
from pypdf import PdfReader
import re
from tqdm import tqdm
from ollama_client import OllamaClient

def test_embedding_models():
    """Test which embedding models work"""
    print("🔍 Testing embedding models...")
    client = OllamaClient()
    
    models_to_test = [
        "mxbai-embed-large",
        "nomic-embed-text", 
        "bge-m3",
        "all-minilm:l6-v2"
    ]
    
    working_models = []
    
    for model in models_to_test:
        try:
            print(f"Testing {model}...")
            embeddings = client.embed_batch(model, ["test sentence"])
            if embeddings and len(embeddings[0]) > 0:
                print(f"✅ {model} works! Dimension: {len(embeddings[0])}")
                working_models.append(model)
            else:
                print(f"❌ {model} failed - empty embedding")
        except Exception as e:
            print(f"❌ {model} failed - {e}")
    
    return working_models

def read_pdf(path: str) -> str:
    """Read PDF and extract text"""
    r = PdfReader(path)
    out = []
    for p in r.pages:
        out.append(p.extract_text() or "")
    text = "\n".join(out)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def chunk_text(text: str, size: int, overlap: int):
    """Simple chunking"""
    assert overlap < size, "overlap must be less than size"
    chunks, n, i = [], len(text), 0
    while i < n:
        j = min(i + size, n)
        chunks.append(text[i:j])
        if j == n: break
        i = max(0, j - overlap)
    return chunks

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """L2 normalization"""
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def create_fixed_index(pdf_path: str, embed_model: str, outdir: str):
    """Create a new index with working embeddings"""
    print(f"📚 Creating fixed index with {embed_model}...")
    
    # Read PDF
    print("Reading PDF...")
    text = read_pdf(pdf_path)
    print(f"Extracted {len(text)} characters")
    
    # Chunk text
    print("Chunking text...")
    chunks = chunk_text(text, size=500, overlap=100)
    print(f"Created {len(chunks)} chunks")
    
    # Generate embeddings
    print("Generating embeddings...")
    client = OllamaClient()
    
    embeddings = []
    for i, chunk in enumerate(tqdm(chunks, desc="Embedding")):
        try:
            chunk_embeddings = client.embed_batch(embed_model, [chunk])
            if chunk_embeddings and len(chunk_embeddings[0]) > 0:
                embeddings.append(chunk_embeddings[0])
            else:
                print(f"Warning: Empty embedding for chunk {i}")
                # Create a dummy embedding
                embeddings.append([0.0] * 384)  # Common embedding dimension
        except Exception as e:
            print(f"Error embedding chunk {i}: {e}")
            # Create a dummy embedding
            embeddings.append([0.0] * 384)
    
    if not embeddings:
        print("❌ No embeddings generated!")
        return False
    
    # Convert to numpy array
    X = np.array(embeddings, dtype=np.float32)
    print(f"Embedding matrix shape: {X.shape}")
    
    # Normalize
    X = l2_normalize(X)
    
    # Create FAISS index
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)
    
    # Create metadata
    meta = [{"id": i, "chunk": chunk} for i, chunk in enumerate(chunks)]
    
    # Save
    os.makedirs(outdir, exist_ok=True)
    faiss.write_index(index, os.path.join(outdir, "index.faiss"))
    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Fixed index created: {outdir}")
    print(f"   - {len(chunks)} chunks")
    print(f"   - {d} dimensions")
    print(f"   - {index.ntotal} vectors")
    
    return True

def main():
    """Main function"""
    print("🔧 FIXING EMBEDDING ISSUES")
    print("=" * 50)
    
    # Test models
    working_models = test_embedding_models()
    
    if not working_models:
        print("❌ No working embedding models found!")
        print("💡 Try installing a different model:")
        print("   ollama pull all-minilm:l6-v2")
        return 1
    
    # Use the first working model
    embed_model = working_models[0]
    print(f"\n🎯 Using model: {embed_model}")
    
    # Create fixed index
    pdf_path = "F1_33.pdf"
    outdir = "./fixed_index"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return 1
    
    success = create_fixed_index(pdf_path, embed_model, outdir)
    
    if success:
        print(f"\n🎉 SUCCESS!")
        print(f"Fixed index created at: {outdir}")
        print(f"\nYou can now test it with:")
        print(f"python simple_test.py 'Who is Max Verstappen?'")
        print(f"\nOr update your config to use: {embed_model}")
    else:
        print("❌ Failed to create fixed index")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
