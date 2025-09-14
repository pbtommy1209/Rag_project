#!/usr/bin/env python3
# quick_start.py — quick setup and test for RAG system
import os, sys, subprocess

def check_ollama():
    """Check if Ollama is running and models are available"""
    print("🔍 Checking Ollama setup...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model.get('name', '') for model in models]
            
            required_models = ['mxbai-embed-large', 'llama3.2']
            missing_models = []
            
            for req_model in required_models:
                if not any(req_model in name for name in model_names):
                    missing_models.append(req_model)
            
            if missing_models:
                print(f"❌ Missing models: {missing_models}")
                print("💡 Install them with:")
                for model in missing_models:
                    print(f"   ollama pull {model}")
                return False
            else:
                print("✅ All required models are available")
                return True
        else:
            print("❌ Ollama not responding")
            return False
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("💡 Start Ollama with: ollama serve")
        return False

def create_simple_index():
    """Create a simple test index"""
    print("\n📚 Creating simple test index...")
    
    if not os.path.exists("F1_33.pdf"):
        print("❌ F1_33.pdf not found in current directory")
        return False
    
    # Create a simple index with just one configuration
    cmd = """python enhanced_ingest.py \\
        --pdf "F1_33.pdf" \\
        --outdir "./simple_index" \\
        --strategy adaptive \\
        --sizes "500" \\
        --overlaps "100" \\
        --embed-model "mxbai-embed-large" \\
        --batch-size 3"""
    
    try:
        print("Running ingestion (this may take a few minutes)...")
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Index created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ingestion failed: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def test_single_question():
    """Test with a single question"""
    print("\n🧪 Testing with a single question...")
    
    # Find the created index
    if os.path.exists("simple_index"):
        indexes = [d for d in os.listdir("simple_index") if os.path.isdir(os.path.join("simple_index", d))]
        if indexes:
            index_path = os.path.join("simple_index", indexes[0])
            
            cmd = f"""python ask_questions.py \\
                --index "{index_path}" \\
                --question "Who is Max Verstappen?" \\
                --embed-model "mxbai-embed-large" \\
                --chat-model "llama3.2" """
            
            try:
                print("Asking: 'Who is Max Verstappen?'")
                result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                print("✅ Question answered successfully!")
                print("\n" + "="*60)
                print("RESULT:")
                print(result.stdout)
                print("="*60)
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ Question test failed: {e}")
                if e.stdout:
                    print("STDOUT:", e.stdout)
                if e.stderr:
                    print("STDERR:", e.stderr)
                return False
    
    print("❌ No index found for testing")
    return False

def main():
    """Quick start workflow"""
    print("🚀 RAG SYSTEM QUICK START")
    print("=" * 60)
    print("This will set up and test your RAG system quickly")
    print("=" * 60)
    
    # Step 1: Check Ollama
    if not check_ollama():
        print("\n❌ Ollama setup incomplete. Please fix the issues above.")
        return 1
    
    # Step 2: Create simple index
    if not create_simple_index():
        print("\n❌ Index creation failed. Please check the errors above.")
        return 1
    
    # Step 3: Test with a question
    if not test_single_question():
        print("\n❌ Question test failed. Please check the errors above.")
        return 1
    
    print("\n🎉 QUICK START COMPLETE!")
    print("=" * 60)
    print("Your RAG system is working! You can now:")
    print("\n1. Ask questions interactively:")
    print("   python ask_questions.py --index ./simple_index/[index_name] --interactive")
    print("\n2. Ask a single question:")
    print("   python ask_questions.py --index ./simple_index/[index_name] --question 'Your question here'")
    print("\n3. Ask multiple questions:")
    print("   python ask_questions.py --index ./simple_index/[index_name] --questions example_questions.json")
    print("\n4. Use the original demo:")
    print("   python run_demo.py --index ./simple_index/[index_name]")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
