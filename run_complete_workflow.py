#!/usr/bin/env python3
# run_complete_workflow.py — complete RAG workflow from ingestion to optimization
import os, sys, argparse, subprocess
from pathlib import Path

def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status"""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def main():
    ap = argparse.ArgumentParser(description="Complete RAG workflow")
    ap.add_argument("--pdf", required=True, help="PDF file path")
    ap.add_argument("--outdir", default="./faiss_indexes", help="Output directory for indexes")
    ap.add_argument("--embed-model", default="mxbai-embed-large", help="Embedding model")
    ap.add_argument("--chat-model", default="llama3.2:3b", help="Chat model")
    ap.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama URL")
    ap.add_argument("--skip-ingestion", action="store_true", help="Skip ingestion step")
    ap.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation step")
    ap.add_argument("--skip-optimization", action="store_true", help="Skip optimization step")
    ap.add_argument("--max-optimization", type=int, default=30, help="Max optimization combinations")
    args = ap.parse_args()

    print("🚀 Starting Complete RAG Workflow")
    print("=" * 60)
    print(f"PDF: {args.pdf}")
    print(f"Output Directory: {args.outdir}")
    print(f"Embedding Model: {args.embed_model}")
    print(f"Chat Model: {args.chat_model}")
    print(f"Ollama URL: {args.ollama_url}")
    print("=" * 60)

    # Step 1: Enhanced Ingestion
    if not args.skip_ingestion:
        print("\n📚 STEP 1: ENHANCED INGESTION")
        print("=" * 60)
        
        # Create output directory
        os.makedirs(args.outdir, exist_ok=True)
        
        # Run enhanced ingestion with multiple strategies
        cmd = f"""python enhanced_ingest.py \\
            --pdf "{args.pdf}" \\
            --outdir "{args.outdir}" \\
            --strategy all \\
            --sizes "200,300,400,500,600,800,1000" \\
            --overlaps "25,50,75,100,125,150,200" \\
            --embed-model "{args.embed_model}" \\
            --ollama-url "{args.ollama_url}" \\
            --batch-size 5"""
        
        if not run_command(cmd, "Running enhanced ingestion with multiple chunking strategies"):
            print("❌ Ingestion failed! Exiting.")
            return 1
    else:
        print("\n⏭️  SKIPPING INGESTION")

    # Step 2: Enhanced Evaluation
    if not args.skip_evaluation:
        print("\n📊 STEP 2: ENHANCED EVALUATION")
        print("=" * 60)
        
        cmd = f"""python enhanced_eval.py \\
            --pdf "{args.pdf}" \\
            --indexes_root "{args.outdir}" \\
            --embed-model "{args.embed_model}" \\
            --chat-model "{args.chat_model}" \\
            --ollama-url "{args.ollama_url}" \\
            --k 5 \\
            --output "{args.outdir}/evaluation_results.json" """
        
        if not run_command(cmd, "Running comprehensive evaluation"):
            print("❌ Evaluation failed! Continuing to optimization...")
    else:
        print("\n⏭️  SKIPPING EVALUATION")

    # Step 3: Parameter Optimization
    if not args.skip_optimization:
        print("\n🎯 STEP 3: PARAMETER OPTIMIZATION")
        print("=" * 60)
        
        cmd = f"""python optimize_params.py \\
            --indexes_root "{args.outdir}" \\
            --embed-model "{args.embed_model}" \\
            --chat-model "{args.chat_model}" \\
            --ollama-url "{args.ollama_url}" \\
            --max-combinations {args.max_optimization} \\
            --output "{args.outdir}/optimization_results.json" """
        
        if not run_command(cmd, "Running parameter optimization"):
            print("❌ Optimization failed! Continuing...")
    else:
        print("\n⏭️  SKIPPING OPTIMIZATION")

    # Step 4: Interactive Demo
    print("\n🎮 STEP 4: INTERACTIVE DEMO")
    print("=" * 60)
    
    # Find the best performing index
    best_index = None
    if os.path.exists(f"{args.outdir}/optimization_results.json"):
        import json
        try:
            with open(f"{args.outdir}/optimization_results.json", 'r') as f:
                opt_results = json.load(f)
            best_config = opt_results['best_configuration']
            best_index = os.path.join(args.outdir, f"adaptive_size{best_config['parameters']['chunk_size']}_overlap{best_config['parameters']['overlap']}")
        except:
            pass
    
    if not best_index or not os.path.exists(best_index):
        # Fallback to a default index
        available_indexes = [d for d in os.listdir(args.outdir) if d.startswith(('adaptive_', 'fixed_', 'semantic_'))]
        if available_indexes:
            best_index = os.path.join(args.outdir, available_indexes[0])
    
    if best_index and os.path.exists(best_index):
        print(f"🎯 Using best performing index: {os.path.basename(best_index)}")
        print(f"📁 Full path: {best_index}")
        
        cmd = f"""python run_demo.py \\
            --index "{best_index}" \\
            --embed-model "{args.embed_model}" \\
            --chat-model "{args.chat_model}" \\
            --ollama-url "{args.ollama_url}" \\
            --k 5"""
        
        print(f"\n🚀 Starting interactive demo...")
        print(f"Command: {cmd}")
        print("=" * 60)
        print("💡 You can now ask questions about your PDF!")
        print("💡 Type 'exit' or press Ctrl+C to quit")
        print("=" * 60)
        
        try:
            subprocess.run(cmd, shell=True)
        except KeyboardInterrupt:
            print("\n👋 Demo ended by user")
    else:
        print("❌ No suitable index found for demo")

    # Summary
    print("\n🎉 WORKFLOW COMPLETE!")
    print("=" * 60)
    print(f"📁 Results saved in: {args.outdir}")
    print("📊 Check evaluation_results.json for performance metrics")
    print("🎯 Check optimization_results.json for best parameters")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
