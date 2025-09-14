#!/usr/bin/env python3
# simple_test.py — simple test to verify Ollama models work with PDF
import os, sys, json, faiss
import numpy as np
from ollama_client import OllamaClient

def test_basic_rag():
    """Test basic RAG functionality"""
    print("🧪 Testing Basic RAG Functionality")
    print("=" * 50)
    
    # Test 1: Load index and metadata
    print("1. Loading FAISS index...")
    index_dir = "./faiss_indexes/size500_overlap100"
    
    try:
        index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"✅ Index loaded: {index.ntotal} vectors")
        print(f"✅ Metadata loaded: {len(meta)} chunks")
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        return False
    
    # Test 2: Test embedding generation
    print("\n2. Testing embedding generation...")
    try:
        client = OllamaClient()
        query = "Who is Max Verstappen?"
        embeddings = client.embed_batch("mxbai-embed-large", [query])
        
        if embeddings and len(embeddings[0]) > 0:
            print(f"✅ Embedding generated: {len(embeddings[0])} dimensions")
        else:
            print("❌ No embedding generated")
            return False
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return False
    
    # Test 3: Test similarity search
    print("\n3. Testing similarity search...")
    try:
        # Normalize embedding
        qv = np.array(embeddings[0], dtype=np.float32).reshape(1, -1)
        norms = np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12
        qv_normalized = qv / norms
        
        # Search
        scores, indices = index.search(qv_normalized, k=3)
        
        print(f"✅ Search completed: {len(indices[0])} results")
        print("Top results:")
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            chunk_text = meta[idx]["chunk"][:100] + "..."
            print(f"  {i+1}. Score: {score:.3f} | {chunk_text}")
    except Exception as e:
        print(f"❌ Error in similarity search: {e}")
        return False
    
    # Test 4: Test chat model
    print("\n4. Testing chat model...")
    try:
        # Get top passage
        top_chunk = meta[indices[0][0]]["chunk"]
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer based only on the provided context."},
            {"role": "user", "content": f"Context: {top_chunk}\n\nQuestion: {query}\n\nAnswer:"}
        ]
        
        response = client.chat("llama3.2", messages, temperature=0.2)
        print(f"✅ Chat response generated: {len(response)} characters")
        print(f"Response: {response[:200]}...")
    except Exception as e:
        print(f"❌ Error in chat: {e}")
        return False
    
    print("\n🎉 All tests passed! Your RAG system is working!")
    return True

def ask_simple_question(question: str):
    """Ask a simple question using the RAG system"""
    print(f"\n🔍 Question: {question}")
    print("-" * 50)
    
    try:
        # Load index
        index_dir = "./faiss_indexes/size500_overlap100"
        index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        # Generate embedding
        client = OllamaClient()
        embeddings = client.embed_batch("mxbai-embed-large", [question])
        
        # Search
        qv = np.array(embeddings[0], dtype=np.float32).reshape(1, -1)
        norms = np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12
        qv_normalized = qv / norms
        
        scores, indices = index.search(qv_normalized, k=5)
        
        # Get top passages
        top_passages = []
        for score, idx in zip(scores[0], indices[0]):
            top_passages.append({
                "score": float(score),
                "chunk": meta[idx]["chunk"]
            })
        
        # Generate answer
        context = "\n\n".join([f"[p{i+1}] {p['chunk']}" for i, p in enumerate(top_passages)])
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer based only on the provided context. Be concise and cite sources with [p1], [p2], etc."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"}
        ]
        
        answer = client.chat("llama3.2", messages, temperature=0.2)
        
        print("📚 Retrieved passages:")
        for i, passage in enumerate(top_passages, 1):
            print(f"\n[{i}] Score: {passage['score']:.3f}")
            print(f"    {passage['chunk'][:150]}...")
        
        print(f"\n💡 Answer:")
        print("=" * 50)
        print(answer)
        print("=" * 50)
        
        return answer
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Ask specific question
        question = " ".join(sys.argv[1:])
        ask_simple_question(question)
    else:
        # Run tests
        if test_basic_rag():
            print("\n" + "=" * 50)
            print("🎮 Now you can ask questions!")
            print("Usage: python simple_test.py 'Your question here'")
            print("\nExample questions:")
            print("• Who is Max Verstappen?")
            print("• Which company sponsors Red Bull?")
            print("• What is the Las Vegas Grand Prix?")

if __name__ == "__main__":
    main()
