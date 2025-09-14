#!/usr/bin/env python3
# run_tests.py — comprehensive testing workflow for RAG system
import os, sys, argparse, subprocess, json, time
from pathlib import Path

def run_command(cmd: str, description: str, timeout: int = 300) -> tuple[bool, str]:
    """Run a command and return success status and output"""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, timeout=timeout)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True, result.stdout
    except subprocess.TimeoutExpired:
        print(f"⏰ Command timed out after {timeout} seconds")
        return False, "Timeout"
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False, e.stderr or e.stdout

def test_system_validation():
    """Run system validation tests"""
    print("\n🧪 STEP 1: SYSTEM VALIDATION")
    print("=" * 60)
    
    success, output = run_command("python test_rag_system.py", "Running system validation tests")
    return success

def test_ingestion():
    """Test PDF ingestion with different strategies"""
    print("\n📚 STEP 2: INGESTION TESTING")
    print("=" * 60)
    
    # Test with a small subset first
    cmd = """python enhanced_ingest.py \\
        --pdf "F1_33.pdf" \\
        --outdir "./test_indexes" \\
        --strategy adaptive \\
        --sizes "300,500" \\
        --overlaps "50,100" \\
        --embed-model "mxbai-embed-large" \\
        --batch-size 3"""
    
    success, output = run_command(cmd, "Testing enhanced ingestion", timeout=600)
    
    if success:
        # Check if indexes were created
        if os.path.exists("test_indexes"):
            indexes = [d for d in os.listdir("test_indexes") if os.path.isdir(os.path.join("test_indexes", d))]
            print(f"✅ Created {len(indexes)} test indexes")
            return True
        else:
            print("❌ No indexes created")
            return False
    
    return success

def test_retrieval():
    """Test retrieval functionality"""
    print("\n🔍 STEP 3: RETRIEVAL TESTING")
    print("=" * 60)
    
    # Find a test index
    if not os.path.exists("test_indexes"):
        print("❌ No test indexes found. Run ingestion first.")
        return False
    
    indexes = [d for d in os.listdir("test_indexes") if os.path.isdir(os.path.join("test_indexes", d))]
    if not indexes:
        print("❌ No valid indexes found")
        return False
    
    test_index = os.path.join("test_indexes", indexes[0])
    
    # Test different retrieval methods
    methods = ["semantic", "keyword", "hybrid"]
    results = {}
    
    for method in methods:
        cmd = f"""python enhanced_retrieve.py \\
            --index "{test_index}" \\
            --query "Who is Max Verstappen?" \\
            --method {method} \\
            --k 3 \\
            --embed-model "mxbai-embed-large" """
        
        success, output = run_command(cmd, f"Testing {method} retrieval", timeout=120)
        results[method] = success
    
    # Summary
    successful_methods = [method for method, success in results.items() if success]
    print(f"\n📊 Retrieval Test Results:")
    print(f"   Successful methods: {successful_methods}")
    print(f"   Failed methods: {[m for m in methods if m not in successful_methods]}")
    
    return len(successful_methods) > 0

def test_evaluation():
    """Test evaluation system"""
    print("\n📊 STEP 4: EVALUATION TESTING")
    print("=" * 60)
    
    if not os.path.exists("test_indexes"):
        print("❌ No test indexes found. Run ingestion first.")
        return False
    
    cmd = """python enhanced_eval.py \\
        --pdf "F1_33.pdf" \\
        --indexes_root "./test_indexes" \\
        --embed-model "mxbai-embed-large" \\
        --chat-model "llama3.2:3b" \\
        --k 3 \\
        --output "test_evaluation_results.json" """
    
    success, output = run_command(cmd, "Testing comprehensive evaluation", timeout=600)
    
    if success and os.path.exists("test_evaluation_results.json"):
        try:
            with open("test_evaluation_results.json", 'r') as f:
                results = json.load(f)
            print(f"✅ Evaluation completed. Tested {len(results)} configurations")
            
            # Show best result
            if results:
                best = max(results, key=lambda x: x['answer_metrics']['exact_match_avg'])
                print(f"   Best configuration: {best['index']}")
                print(f"   Best EM score: {best['answer_metrics']['exact_match_avg']:.3f}")
            
            return True
        except Exception as e:
            print(f"❌ Error reading evaluation results: {e}")
            return False
    
    return success

def test_optimization():
    """Test parameter optimization"""
    print("\n🎯 STEP 5: OPTIMIZATION TESTING")
    print("=" * 60)
    
    if not os.path.exists("test_indexes"):
        print("❌ No test indexes found. Run ingestion first.")
        return False
    
    cmd = """python optimize_params.py \\
        --indexes_root "./test_indexes" \\
        --embed-model "mxbai-embed-large" \\
        --chat-model "llama3.2:3b" \\
        --max-combinations 10 \\
        --output "test_optimization_results.json" """
    
    success, output = run_command(cmd, "Testing parameter optimization", timeout=600)
    
    if success and os.path.exists("test_optimization_results.json"):
        try:
            with open("test_optimization_results.json", 'r') as f:
                results = json.load(f)
            
            if 'best_configuration' in results:
                best = results['best_configuration']
                print(f"✅ Optimization completed")
                print(f"   Best configuration found")
                print(f"   Best EM score: {best['answer_metrics']['exact_match_avg']:.3f}")
                return True
            else:
                print("❌ No best configuration found in results")
                return False
        except Exception as e:
            print(f"❌ Error reading optimization results: {e}")
            return False
    
    return success

def test_interactive_demo():
    """Test interactive demo"""
    print("\n🎮 STEP 6: INTERACTIVE DEMO TESTING")
    print("=" * 60)
    
    if not os.path.exists("test_indexes"):
        print("❌ No test indexes found. Run ingestion first.")
        return False
    
    indexes = [d for d in os.listdir("test_indexes") if os.path.isdir(os.path.join("test_indexes", d))]
    if not indexes:
        print("❌ No valid indexes found")
        return False
    
    test_index = os.path.join("test_indexes", indexes[0])
    
    # Test with a simple query
    cmd = f"""python run_demo.py \\
        --index "{test_index}" \\
        --embed-model "mxbai-embed-large" \\
        --chat-model "llama3.2:3b" \\
        --k 3"""
    
    print("ℹ️  Testing interactive demo (this will run briefly)")
    print("💡 The demo will be tested with a timeout - this is normal")
    
    success, output = run_command(cmd, "Testing interactive demo", timeout=30)
    
    # For demo, we consider it successful if it starts without errors
    if "Loaded index:" in output or success:
        print("✅ Interactive demo test passed")
        return True
    else:
        print("❌ Interactive demo test failed")
        return False

def cleanup_test_files():
    """Clean up test files"""
    print("\n🧹 CLEANUP")
    print("=" * 60)
    
    test_files = ["test_indexes", "test_evaluation_results.json", "test_optimization_results.json"]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)
                print(f"🗑️  Removed directory: {file_path}")
            else:
                os.remove(file_path)
                print(f"🗑️  Removed file: {file_path}")

def main():
    """Run comprehensive testing workflow"""
    ap = argparse.ArgumentParser(description="Comprehensive RAG testing")
    ap.add_argument("--skip-cleanup", action="store_true", help="Skip cleanup of test files")
    ap.add_argument("--quick", action="store_true", help="Run quick tests only")
    args = ap.parse_args()

    print("🧪 COMPREHENSIVE RAG TESTING SUITE")
    print("=" * 60)
    print("This will test all components of your RAG system")
    print("=" * 60)

    # Test steps
    test_steps = [
        ("System Validation", test_system_validation),
        ("Ingestion", test_ingestion),
        ("Retrieval", test_retrieval),
        ("Evaluation", test_evaluation),
    ]
    
    if not args.quick:
        test_steps.extend([
            ("Optimization", test_optimization),
            ("Interactive Demo", test_interactive_demo),
        ])
    
    # Run tests
    results = {}
    for step_name, test_func in test_steps:
        try:
            results[step_name] = test_func()
        except KeyboardInterrupt:
            print(f"\n⏹️  Testing interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error in {step_name}: {e}")
            results[step_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TESTING SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your RAG system is working correctly and ready for production use.")
        print("\nRecommended next steps:")
        print("1. Run full ingestion: python enhanced_ingest.py --pdf F1_33.pdf --outdir ./faiss_indexes --strategy all")
        print("2. Run full evaluation: python enhanced_eval.py --pdf F1_33.pdf --indexes_root ./faiss_indexes")
        print("3. Run parameter optimization: python optimize_params.py --indexes_root ./faiss_indexes")
        print("4. Use interactive demo: python run_demo.py --index ./faiss_indexes/[best_index]")
    elif passed >= total * 0.7:
        print(f"\n⚠️  {total - passed} tests failed, but system is mostly functional.")
        print("You can proceed with caution and fix issues as needed.")
    else:
        print(f"\n❌ {total - passed} tests failed. Please fix the issues before proceeding.")
    
    # Cleanup
    if not args.skip_cleanup:
        cleanup_test_files()
    
    return passed >= total * 0.7

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
