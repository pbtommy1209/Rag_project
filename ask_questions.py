#!/usr/bin/env python3
# ask_questions.py — simple script to ask questions using your RAG system
import os, sys, argparse, json
from typing import List, Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def ask_question(index_dir: str, question: str, embed_model: str, chat_model: str, ollama_url: str, k: int = 5):
    """Ask a single question using the RAG system"""
    try:
        from enhanced_retrieve import retrieve_enhanced
        from answer import rag_answer
        
        print(f"🔍 Question: {question}")
        print("-" * 60)
        
        # Step 1: Retrieve relevant passages
        print("📚 Retrieving relevant passages...")
        hits = retrieve_enhanced(
            index_dir, question, k=k, 
            embed_model=embed_model, ollama_url=ollama_url, method="hybrid"
        )
        
        print(f"✅ Found {len(hits)} relevant passages:")
        for i, hit in enumerate(hits, 1):
            print(f"\n[{i}] Score: {hit['score']:.3f}")
            print(f"    {hit['chunk'][:200]}...")
        
        # Step 2: Generate answer using retrieved passages
        print(f"\n🤖 Generating answer using {chat_model}...")
        answer = rag_answer(question, hits, chat_model, ollama_url)
        
        print(f"\n💡 Answer:")
        print("=" * 60)
        print(answer)
        print("=" * 60)
        
        return {
            "question": question,
            "answer": answer,
            "passages": hits,
            "num_passages": len(hits)
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def interactive_qa(index_dir: str, embed_model: str, chat_model: str, ollama_url: str, k: int = 5):
    """Interactive question-answering session"""
    print("🎮 Interactive RAG Question-Answering")
    print("=" * 60)
    print(f"Index: {os.path.basename(index_dir)}")
    print(f"Embedding Model: {embed_model}")
    print(f"Chat Model: {chat_model}")
    print("=" * 60)
    print("💡 Type your questions below. Type 'quit' or 'exit' to stop.")
    print("💡 Type 'help' for example questions.")
    print("-" * 60)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif question.lower() == 'help':
                print("\n📝 Example questions you can ask:")
                print("   • Who is Max Verstappen?")
                print("   • Which company sponsors Red Bull?")
                print("   • What is the Las Vegas Grand Prix?")
                print("   • How much is Verstappen's contract worth?")
                print("   • Why did Verstappen complain in Austin?")
                print("   • What boosted F1's popularity in the US?")
                continue
            elif not question:
                print("Please enter a question.")
                continue
            
            # Ask the question
            result = ask_question(index_dir, question, embed_model, chat_model, ollama_url, k)
            
            if result:
                print(f"\n✅ Question answered successfully!")
            else:
                print(f"\n❌ Failed to answer question.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def batch_questions(index_dir: str, questions: List[str], embed_model: str, chat_model: str, ollama_url: str, k: int = 5):
    """Ask multiple questions in batch"""
    print(f"📋 Batch Question-Answering ({len(questions)} questions)")
    print("=" * 60)
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n🔍 Question {i}/{len(questions)}")
        result = ask_question(index_dir, question, embed_model, chat_model, ollama_url, k)
        if result:
            results.append(result)
        
        # Brief pause between questions
        if i < len(questions):
            print("\n" + "-" * 60)
    
    # Summary
    print(f"\n📊 Batch Results Summary")
    print("=" * 60)
    print(f"Total questions: {len(questions)}")
    print(f"Successfully answered: {len(results)}")
    print(f"Success rate: {len(results)/len(questions)*100:.1f}%")
    
    return results

def main():
    """Main function for asking questions"""
    ap = argparse.ArgumentParser(description="Ask questions using your RAG system")
    ap.add_argument("--index", required=True, help="FAISS index directory")
    ap.add_argument("--question", help="Single question to ask")
    ap.add_argument("--questions", help="JSON file with list of questions")
    ap.add_argument("--interactive", action="store_true", help="Start interactive mode")
    ap.add_argument("--embed-model", default="mxbai-embed-large:latest", help="Embedding model")
    ap.add_argument("--chat-model", default="llama3.2", help="Chat model")
    ap.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama URL")
    ap.add_argument("--k", type=int, default=5, help="Number of passages to retrieve")
    ap.add_argument("--output", help="Output JSON file for results")
    args = ap.parse_args()

    # Check if index exists
    if not os.path.exists(args.index):
        print(f"❌ Index directory not found: {args.index}")
        print("💡 Make sure you've run ingestion first:")
        print("   python enhanced_ingest.py --pdf F1_33.pdf --outdir ./faiss_indexes")
        return 1

    # Check if index has required files
    required_files = ["index.faiss", "meta.json"]
    for file in required_files:
        if not os.path.exists(os.path.join(args.index, file)):
            print(f"❌ Index file missing: {file}")
            return 1

    print("🚀 RAG Question-Answering System")
    print("=" * 60)
    print(f"Index: {args.index}")
    print(f"Embedding Model: {args.embed_model}")
    print(f"Chat Model: {args.chat_model}")
    print(f"Retrieval Method: Hybrid (semantic + keyword)")
    print(f"Top-K: {args.k}")
    print("=" * 60)

    results = []

    if args.interactive:
        # Interactive mode
        interactive_qa(args.index, args.embed_model, args.chat_model, args.ollama_url, args.k)
        
    elif args.question:
        # Single question
        result = ask_question(args.index, args.question, args.embed_model, args.chat_model, args.ollama_url, args.k)
        if result:
            results.append(result)
            
    elif args.questions:
        # Batch questions from file
        try:
            with open(args.questions, 'r') as f:
                questions_data = json.load(f)
            
            if isinstance(questions_data, list):
                questions = questions_data
            elif isinstance(questions_data, dict) and 'questions' in questions_data:
                questions = questions_data['questions']
            else:
                print("❌ Invalid questions file format. Expected list of questions or {'questions': [...]}")
                return 1
            
            results = batch_questions(args.index, questions, args.embed_model, args.chat_model, args.ollama_url, args.k)
            
        except Exception as e:
            print(f"❌ Error reading questions file: {e}")
            return 1
            
    else:
        # Default: ask some example questions
        example_questions = [
            "Who is Max Verstappen?",
            "Which company sponsors the Red Bull team?",
            "What is the Las Vegas Grand Prix?",
            "How much is Verstappen's contract worth?",
            "Why did Verstappen complain during the Austin race?"
        ]
        
        print("📝 No specific question provided. Asking example questions...")
        results = batch_questions(args.index, example_questions, args.embed_model, args.chat_model, args.ollama_url, args.k)

    # Save results
    if args.output and results:
        try:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {args.output}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
