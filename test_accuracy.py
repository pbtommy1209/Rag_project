#!/usr/bin/env python3
# test_accuracy.py — focused accuracy testing for RAG system
import os, sys, argparse, json
from typing import List, Dict
import time

def create_test_questions() -> List[Dict]:
    """Create a comprehensive set of test questions for accuracy testing"""
    return [
        # Factual questions (should be easy to answer correctly)
        {
            "question": "Who is the main subject of this article?",
            "expected_answer": "Max Verstappen",
            "question_type": "factual",
            "difficulty": "easy"
        },
        {
            "question": "Which company sponsors the Red Bull team?",
            "expected_answer": "Oracle",
            "question_type": "factual", 
            "difficulty": "easy"
        },
        {
            "question": "What is the name of the Las Vegas Grand Prix?",
            "expected_answer": "Las Vegas Grand Prix",
            "question_type": "factual",
            "difficulty": "easy"
        },
        {
            "question": "Which Netflix series boosted F1 popularity in the US?",
            "expected_answer": "Drive to Survive",
            "question_type": "factual",
            "difficulty": "medium"
        },
        {
            "question": "Who did Verstappen battle for the championship in 2021?",
            "expected_answer": "Lewis Hamilton",
            "question_type": "factual",
            "difficulty": "medium"
        },
        
        # Numerical questions
        {
            "question": "What is Verstappen's approximate annual contract value in millions?",
            "expected_answer": "50",
            "question_type": "numerical",
            "difficulty": "medium"
        },
        {
            "question": "How many world championships has Verstappen won?",
            "expected_answer": "three",
            "question_type": "numerical",
            "difficulty": "easy"
        },
        
        # Reasoning questions (require understanding context)
        {
            "question": "Why did Verstappen complain during the Austin race?",
            "expected_answer": "brakes",
            "question_type": "reasoning",
            "difficulty": "hard"
        },
        {
            "question": "What factors contributed to F1's growth in the United States?",
            "expected_answer": "Drive to Survive",
            "question_type": "reasoning",
            "difficulty": "hard"
        },
        
        # Comparative questions
        {
            "question": "Which is more valuable: Verstappen's current contract or Hamilton's legacy?",
            "expected_answer": "Verstappen's contract",
            "question_type": "comparative",
            "difficulty": "hard"
        }
    ]

def test_single_question(index_dir: str, question: Dict, embed_model: str, chat_model: str, ollama_url: str) -> Dict:
    """Test a single question and return detailed results"""
    try:
        from enhanced_retrieve import retrieve_enhanced
        from answer import rag_answer
        
        # Retrieve relevant passages
        hits = retrieve_enhanced(
            index_dir, question["question"], k=5, 
            embed_model=embed_model, ollama_url=ollama_url, method="hybrid"
        )
        
        # Generate answer
        answer = rag_answer(question["question"], hits, chat_model, ollama_url)
        
        # Check if expected answer is in the generated answer
        expected = question["expected_answer"].lower()
        generated = answer.lower()
        
        # Simple exact match
        exact_match = expected in generated
        
        # Check if expected answer is in retrieved passages
        passage_contains = any(expected in hit["chunk"].lower() for hit in hits)
        
        # Calculate retrieval relevance (simplified)
        retrieval_score = max([hit["score"] for hit in hits]) if hits else 0
        
        return {
            "question": question["question"],
            "expected_answer": question["expected_answer"],
            "generated_answer": answer,
            "question_type": question["question_type"],
            "difficulty": question["difficulty"],
            "exact_match": exact_match,
            "passage_contains_answer": passage_contains,
            "retrieval_score": retrieval_score,
            "num_passages": len(hits),
            "success": exact_match
        }
        
    except Exception as e:
        return {
            "question": question["question"],
            "expected_answer": question["expected_answer"],
            "generated_answer": f"ERROR: {str(e)}",
            "question_type": question["question_type"],
            "difficulty": question["difficulty"],
            "exact_match": False,
            "passage_contains_answer": False,
            "retrieval_score": 0,
            "num_passages": 0,
            "success": False,
            "error": str(e)
        }

def calculate_accuracy_metrics(results: List[Dict]) -> Dict:
    """Calculate comprehensive accuracy metrics"""
    total = len(results)
    if total == 0:
        return {}
    
    # Overall accuracy
    exact_matches = sum(1 for r in results if r["exact_match"])
    overall_accuracy = exact_matches / total
    
    # Accuracy by question type
    type_accuracy = {}
    for q_type in ["factual", "numerical", "reasoning", "comparative"]:
        type_results = [r for r in results if r["question_type"] == q_type]
        if type_results:
            type_matches = sum(1 for r in type_results if r["exact_match"])
            type_accuracy[q_type] = type_matches / len(type_results)
    
    # Accuracy by difficulty
    difficulty_accuracy = {}
    for difficulty in ["easy", "medium", "hard"]:
        diff_results = [r for r in results if r["difficulty"] == difficulty]
        if diff_results:
            diff_matches = sum(1 for r in diff_results if r["exact_match"])
            difficulty_accuracy[difficulty] = diff_matches / len(diff_results)
    
    # Retrieval quality
    avg_retrieval_score = sum(r["retrieval_score"] for r in results) / total
    passages_with_answer = sum(1 for r in results if r["passage_contains_answer"])
    retrieval_quality = passages_with_answer / total
    
    # Average answer length
    avg_answer_length = sum(len(r["generated_answer"]) for r in results) / total
    
    return {
        "overall_accuracy": overall_accuracy,
        "exact_matches": exact_matches,
        "total_questions": total,
        "type_accuracy": type_accuracy,
        "difficulty_accuracy": difficulty_accuracy,
        "avg_retrieval_score": avg_retrieval_score,
        "retrieval_quality": retrieval_quality,
        "avg_answer_length": avg_answer_length
    }

def print_accuracy_report(results: List[Dict], metrics: Dict):
    """Print a comprehensive accuracy report"""
    print("\n" + "="*80)
    print("📊 RAG ACCURACY TEST REPORT")
    print("="*80)
    
    # Overall metrics
    print(f"\n🎯 OVERALL PERFORMANCE")
    print("-" * 40)
    print(f"Total Questions: {metrics['total_questions']}")
    print(f"Correct Answers: {metrics['exact_matches']}")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.1%}")
    print(f"Retrieval Quality: {metrics['retrieval_quality']:.1%}")
    print(f"Avg Retrieval Score: {metrics['avg_retrieval_score']:.3f}")
    print(f"Avg Answer Length: {metrics['avg_answer_length']:.0f} chars")
    
    # Performance by question type
    print(f"\n📋 ACCURACY BY QUESTION TYPE")
    print("-" * 40)
    for q_type, accuracy in metrics['type_accuracy'].items():
        count = len([r for r in results if r['question_type'] == q_type])
        print(f"{q_type.capitalize():<12} {accuracy:.1%} ({count} questions)")
    
    # Performance by difficulty
    print(f"\n🎚️ ACCURACY BY DIFFICULTY")
    print("-" * 40)
    for difficulty, accuracy in metrics['difficulty_accuracy'].items():
        count = len([r for r in results if r['difficulty'] == difficulty])
        print(f"{difficulty.capitalize():<8} {accuracy:.1%} ({count} questions)")
    
    # Detailed results
    print(f"\n📝 DETAILED RESULTS")
    print("-" * 80)
    print(f"{'Q#':<3} {'Type':<10} {'Diff':<6} {'Correct':<8} {'Answer Preview'}")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        status = "✅" if result["exact_match"] else "❌"
        preview = result["generated_answer"][:50] + "..." if len(result["generated_answer"]) > 50 else result["generated_answer"]
        print(f"{i:<3} {result['question_type']:<10} {result['difficulty']:<6} {status:<8} {preview}")
    
    # Failed questions analysis
    failed = [r for r in results if not r["exact_match"]]
    if failed:
        print(f"\n❌ FAILED QUESTIONS ANALYSIS")
        print("-" * 40)
        for result in failed:
            print(f"\nQ: {result['question']}")
            print(f"Expected: {result['expected_answer']}")
            print(f"Got: {result['generated_answer'][:100]}...")
            if "error" in result:
                print(f"Error: {result['error']}")

def main():
    """Run accuracy testing"""
    ap = argparse.ArgumentParser(description="Test RAG system accuracy")
    ap.add_argument("--index", required=True, help="FAISS index directory")
    ap.add_argument("--embed-model", default="mxbai-embed-large", help="Embedding model")
    ap.add_argument("--chat-model", default="llama3.2:3b", help="Chat model")
    ap.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama URL")
    ap.add_argument("--output", help="Output JSON file for results")
    ap.add_argument("--questions", type=int, help="Number of questions to test (default: all)")
    args = ap.parse_args()

    print("🧪 RAG ACCURACY TESTING")
    print("=" * 60)
    print(f"Index: {args.index}")
    print(f"Embedding Model: {args.embed_model}")
    print(f"Chat Model: {args.chat_model}")
    print("=" * 60)

    # Get test questions
    all_questions = create_test_questions()
    if args.questions:
        questions = all_questions[:args.questions]
        print(f"Testing {len(questions)} questions (limited from {len(all_questions)})")
    else:
        questions = all_questions
        print(f"Testing all {len(questions)} questions")

    # Run tests
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n🔍 Testing question {i}/{len(questions)}: {question['question'][:50]}...")
        
        result = test_single_question(
            args.index, question, args.embed_model, args.chat_model, args.ollama_url
        )
        results.append(result)
        
        # Brief pause to avoid overwhelming the system
        time.sleep(1)
    
    # Calculate metrics
    metrics = calculate_accuracy_metrics(results)
    
    # Print report
    print_accuracy_report(results, metrics)
    
    # Save results
    if args.output:
        output_data = {
            "test_config": {
                "index": args.index,
                "embed_model": args.embed_model,
                "chat_model": args.chat_model,
                "ollama_url": args.ollama_url
            },
            "metrics": metrics,
            "detailed_results": results
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
    
    # Return success based on accuracy
    success = metrics['overall_accuracy'] >= 0.6  # 60% accuracy threshold
    print(f"\n{'🎉' if success else '⚠️'} Test {'PASSED' if success else 'FAILED'} (threshold: 60%)")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
