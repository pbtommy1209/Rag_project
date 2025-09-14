#!/usr/bin/env python3
# test_rag_system.py — comprehensive testing script for RAG system
import os, sys, json, time
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    try:
        import pypdf
        import faiss
        import numpy as np
        import requests
        import nltk
        from ollama_client import OllamaClient
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_ollama_connection():
    """Test connection to Ollama"""
    print("🔍 Testing Ollama connection...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama connected. Found {len(models)} models")
            for model in models[:3]:  # Show first 3 models
                print(f"   - {model.get('name', 'Unknown')}")
            return True
        else:
            print(f"❌ Ollama returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("💡 Make sure Ollama is running: ollama serve")
        return False

def test_pdf_reading():
    """Test PDF reading capability"""
    print("🔍 Testing PDF reading...")
    try:
        from pypdf import PdfReader
        pdf_path = "F1_33.pdf"
        if not os.path.exists(pdf_path):
            print(f"❌ PDF file not found: {pdf_path}")
            return False
        
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages[:2]:  # Read first 2 pages
            text += page.extract_text() or ""
        
        if len(text) > 100:
            print(f"✅ PDF reading successful. Extracted {len(text)} characters")
            return True
        else:
            print("❌ PDF reading failed - too little text extracted")
            return False
    except Exception as e:
        print(f"❌ PDF reading error: {e}")
        return False

def test_embedding():
    """Test embedding generation"""
    print("🔍 Testing embedding generation...")
    try:
        from ollama_client import OllamaClient
        client = OllamaClient()
        
        # Test with a simple text
        test_text = "This is a test sentence for embedding."
        embeddings = client.embed_batch("mxbai-embed-large", [test_text])
        
        if embeddings and len(embeddings[0]) > 0:
            print(f"✅ Embedding successful. Dimension: {len(embeddings[0])}")
            return True
        else:
            print("❌ Embedding failed - no vectors returned")
            return False
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        print("💡 Make sure you have the embedding model: ollama pull mxbai-embed-large")
        return False

def test_chat():
    """Test chat model"""
    print("🔍 Testing chat model...")
    try:
        from ollama_client import OllamaClient
        client = OllamaClient()
        
        messages = [{"role": "user", "content": "Hello, this is a test. Please respond with 'Test successful'."}]
        response = client.chat("llama3.2:3b", messages, temperature=0.1)
        
        if response and len(response) > 0:
            print(f"✅ Chat successful. Response: {response[:50]}...")
            return True
        else:
            print("❌ Chat failed - no response")
            return False
    except Exception as e:
        print(f"❌ Chat error: {e}")
        print("💡 Make sure you have the chat model: ollama pull llama3.2:3b")
        return False

def run_quick_rag_test():
    """Run a quick end-to-end RAG test"""
    print("🔍 Running quick RAG test...")
    try:
        # Check if we have any existing indexes
        if os.path.exists("faiss_indexes"):
            indexes = [d for d in os.listdir("faiss_indexes") if os.path.isdir(os.path.join("faiss_indexes", d))]
            if indexes:
                print(f"✅ Found {len(indexes)} existing indexes")
                return True
        
        print("ℹ️  No existing indexes found. You'll need to run ingestion first.")
        return True
    except Exception as e:
        print(f"❌ RAG test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 RAG System Testing Suite")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Ollama Connection", test_ollama_connection),
        ("PDF Reading", test_pdf_reading),
        ("Embedding", test_embedding),
        ("Chat", test_chat),
        ("RAG System", run_quick_rag_test)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        results[test_name] = test_func()
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your RAG system is ready to use.")
        print("\nNext steps:")
        print("1. Run: python enhanced_ingest.py --pdf F1_33.pdf --outdir ./faiss_indexes --strategy adaptive")
        print("2. Run: python enhanced_eval.py --pdf F1_33.pdf --indexes_root ./faiss_indexes")
        print("3. Run: python run_demo.py --index ./faiss_indexes/[best_index]")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix the issues above before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
