#!/usr/bin/env python3
# test_ollama_models.py — comprehensive test for Ollama models in RAG project
import os, sys, json, time
from typing import List, Dict

def test_ollama_connection():
    """Test basic connection to Ollama"""
    print("🔍 Testing Ollama connection...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama connected successfully")
            print(f"   Found {len(models)} models:")
            for model in models:
                print(f"   - {model.get('name', 'Unknown')} ({model.get('size', 0) / 1024**3:.1f}GB)")
            return True, models
        else:
            print(f"❌ Ollama returned status {response.status_code}")
            return False, []
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("💡 Make sure Ollama is running: ollama serve")
        return False, []

def test_embedding_model():
    """Test embedding model functionality"""
    print("\n🔍 Testing embedding model (mxbai-embed-large:latest)...")
    try:
        from ollama_client import OllamaClient
        client = OllamaClient()
        
        # Test with simple text
        test_texts = [
            "This is a test sentence for embedding.",
            "Max Verstappen is a Formula 1 driver.",
            "Oracle sponsors the Red Bull team."
        ]
        
        print("   Generating embeddings...")
        start_time = time.time()
    embeddings = client.embed_batch("mxbai-embed-large:latest", test_texts)
        end_time = time.time()
        
        if embeddings and len(embeddings) == len(test_texts):
            print(f"✅ Embedding model working correctly")
            print(f"   Generated {len(embeddings)} embeddings")
            print(f"   Embedding dimension: {len(embeddings[0])}")
            print(f"   Time taken: {end_time - start_time:.2f} seconds")
            
            # Test embedding quality (should be similar for similar texts)
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Test similarity between similar sentences
            similar_texts = [
                "Max Verstappen is a Formula 1 driver.",
                "Verstappen drives in Formula 1."
            ]
            similar_embeddings = client.embed_batch("mxbai-embed-large:latest", similar_texts)
            similarity = cosine_similarity([similar_embeddings[0]], [similar_embeddings[1]])[0][0]
            print(f"   Similarity test: {similarity:.3f} (should be > 0.7)")
            
            return True, embeddings
        else:
            print("❌ Embedding model failed - incorrect number of embeddings")
            return False, []
            
    except Exception as e:
        print(f"❌ Embedding model error: {e}")
        return False, []

def test_chat_model():
    """Test chat model functionality"""
    print("\n🔍 Testing chat model (llama3.2)...")
    try:
        from ollama_client import OllamaClient
        client = OllamaClient()
        
        # Test with simple question
        test_messages = [
            {"role": "user", "content": "Hello! Please respond with exactly 'Test successful' to confirm the model is working."}
        ]
        
        print("   Generating chat response...")
        start_time = time.time()
        response = client.chat("llama3.2:latest", test_messages, temperature=0.1)
        end_time = time.time()
        
        if response and len(response) > 0:
            print(f"✅ Chat model working correctly")
            print(f"   Response: {response[:100]}...")
            print(f"   Response length: {len(response)} characters")
            print(f"   Time taken: {end_time - start_time:.2f} seconds")
            
            # Test with RAG-style prompt
            rag_messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer based only on the provided context."},
                {"role": "user", "content": "Context: Max Verstappen is a Formula 1 driver for Red Bull Racing.\n\nQuestion: Who is Max Verstappen?"}
            ]
            
            print("   Testing RAG-style prompt...")
            rag_response = client.chat("", rag_messages, temperature=0.2)
            print(f"   RAG response: {rag_response[:100]}...")
            
            return True, response
        else:
            print("❌ Chat model failed - no response generated")
            return False, ""
            
    except Exception as e:
        print(f"❌ Chat model error: {e}")
        return False, ""

def test_rag_integration():
    """Test RAG integration with both models"""
    print("\n🔍 Testing RAG integration...")
    try:
        from ollama_client import OllamaClient
        
        # Simulate RAG workflow
        client = OllamaClient()
        
        # 1. Test embedding generation for documents
        print("   Step 1: Testing document embedding...")
        document_text = "Max Verstappen is a Dutch Formula 1 driver who races for Red Bull Racing. He has won multiple world championships."
    doc_embedding = client.embed_batch("mxbai-embed-large:latest", [document_text])[0]
        print(f"   ✅ Document embedded successfully (dim: {len(doc_embedding)})")
        
        # 2. Test query embedding
        print("   Step 2: Testing query embedding...")
        query = "Who is Max Verstappen?"
    query_embedding = client.embed_batch("mxbai-embed-large:latest", [query])[0]
        print(f"   ✅ Query embedded successfully (dim: {len(query_embedding)})")
        
        # 3. Test similarity calculation
        print("   Step 3: Testing similarity calculation...")
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        doc_vec = np.array(doc_embedding).reshape(1, -1)
        query_vec = np.array(query_embedding).reshape(1, -1)
        similarity = cosine_similarity(query_vec, doc_vec)[0][0]
        print(f"   ✅ Similarity calculated: {similarity:.3f}")
        
        # 4. Test RAG answer generation
        print("   Step 4: Testing RAG answer generation...")
        rag_messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer based only on the provided context. Be concise."},
            {"role": "user", "content": f"Context: {document_text}\n\nQuestion: {query}\n\nAnswer:"}
        ]
        
        rag_answer = client.chat("llama3.2", rag_messages, temperature=0.2)
        print(f"   ✅ RAG answer generated: {rag_answer[:100]}...")
        
        return True, {
            "document_embedding": doc_embedding,
            "query_embedding": query_embedding,
            "similarity": similarity,
            "rag_answer": rag_answer
        }
        
    except Exception as e:
        print(f"❌ RAG integration error: {e}")
        return False, {}

def test_model_performance():
    """Test model performance metrics"""
    print("\n🔍 Testing model performance...")
    try:
        from ollama_client import OllamaClient
        client = OllamaClient()
        
        # Test embedding performance
        print("   Testing embedding performance...")
        test_texts = ["Test sentence " + str(i) for i in range(5)]
        
        start_time = time.time()
    embeddings = client.embed_batch("mxbai-embed-large:latest", test_texts)
        embedding_time = time.time() - start_time
        
        print(f"   ✅ Embedding performance:")
        print(f"      - 5 texts in {embedding_time:.2f} seconds")
        print(f"      - {embedding_time/5:.2f} seconds per text")
        print(f"      - {5/embedding_time:.1f} texts per second")
        
        # Test chat performance
        print("   Testing chat performance...")
        test_message = [{"role": "user", "content": "Say 'Hello' and nothing else."}]
        
        start_time = time.time()
        response = client.chat("llama3.2", test_message, temperature=0.1)
        chat_time = time.time() - start_time
        
        print(f"   ✅ Chat performance:")
        print(f"      - Response in {chat_time:.2f} seconds")
        print(f"      - Response length: {len(response)} characters")
        print(f"      - {len(response)/chat_time:.1f} characters per second")
        
        return True, {
            "embedding_time": embedding_time,
            "chat_time": chat_time,
            "embedding_throughput": 5/embedding_time,
            "chat_throughput": len(response)/chat_time
        }
        
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        return False, {}

def test_model_quality():
    """Test model quality with sample questions"""
    print("\n🔍 Testing model quality...")
    try:
        from ollama_client import OllamaClient
        client = OllamaClient()
        
        # Test embedding quality
        print("   Testing embedding quality...")
        similar_pairs = [
            ("Max Verstappen is a Formula 1 driver", "Verstappen races in F1"),
            ("Oracle sponsors Red Bull", "Red Bull is sponsored by Oracle"),
            ("Las Vegas Grand Prix", "F1 race in Las Vegas")
        ]
        
        embedding_quality_scores = []
        for text1, text2 in similar_pairs:
            emb1 = client.embed_batch("mxbai-embed-large:latest", [text1])[0]
            emb2 = client.embed_batch("mxbai-embed-large:latest", [text2])[0]
            
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            embedding_quality_scores.append(similarity)
            print(f"      '{text1[:30]}...' vs '{text2[:30]}...': {similarity:.3f}")
        
        avg_embedding_quality = sum(embedding_quality_scores) / len(embedding_quality_scores)
        print(f"   ✅ Average embedding similarity: {avg_embedding_quality:.3f}")
        
        # Test chat quality
        print("   Testing chat quality...")
        quality_questions = [
            "What is 2+2?",
            "Who is the current president of the United States?",
            "Explain what Formula 1 is in one sentence."
        ]
        
        chat_responses = []
        for question in quality_questions:
            messages = [{"role": "user", "content": question}]
            response = client.chat("llama3.2", messages, temperature=0.3)
            chat_responses.append(response)
            print(f"      Q: {question}")
            print(f"      A: {response[:80]}...")
        
        return True, {
            "embedding_quality": avg_embedding_quality,
            "chat_responses": chat_responses
        }
        
    except Exception as e:
        print(f"❌ Quality test error: {e}")
        return False, {}

def main():
    """Run comprehensive Ollama model testing"""
    print("🧪 OLLAMA MODEL TESTING SUITE")
    print("=" * 60)
    print("This will test if your Ollama models work correctly with the RAG project")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Ollama Connection", test_ollama_connection),
        ("Embedding Model", test_embedding_model),
        ("Chat Model", test_chat_model),
        ("RAG Integration", test_rag_integration),
        ("Model Performance", test_model_performance),
        ("Model Quality", test_model_quality)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        try:
            success, data = test_func()
            results[test_name] = {"success": success, "data": data}
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results[test_name] = {"success": False, "data": None, "error": str(e)}
        
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r["success"])
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if not result["success"] and "error" in result:
            print(f"   Error: {result['error']}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your Ollama models are working perfectly with the RAG project.")
        print("\nYou can now proceed with:")
        print("1. python enhanced_ingest.py --pdf F1_33.pdf --outdir ./faiss_indexes")
        print("2. python test_accuracy.py --index ./faiss_indexes/[index_name]")
        print("3. python run_demo.py --index ./faiss_indexes/[index_name]")
    elif passed >= total * 0.8:
        print(f"\n⚠️  {total - passed} tests failed, but core functionality works.")
        print("You can proceed with caution.")
    else:
        print(f"\n❌ {total - passed} tests failed. Please fix the issues before proceeding.")
        print("\nCommon solutions:")
        print("- Make sure Ollama is running: ollama serve")
        print("- Check if models are installed: ollama list")
    print("- Install missing models: ollama pull mxbai-embed-large:latest")
        print("- Install missing models: ollama pull llama3.2")
    
    return passed >= total * 0.8

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
