# run_demo.py — interactive end-to-end: retrieve + answer via Ollama
import argparse, json, os, sys
from retrieve import retrieve
from answer import rag_answer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--embed-model", type=str, default="mxbai-embed-large")
    ap.add_argument("--chat-model", type=str, default="llama3.2:3b")
    ap.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    print("Loaded index:", args.index)
    while True:
        try:
            q = input("\nQuery (blank to exit): ").strip()
        except EOFError:
            break
        if not q:
            break
        hits = retrieve(args.index, q, args.k, args.embed_model, args.ollama_url)
        for h in hits:
            print(f'[{h["rank"]}] {h["chunk"][:280].replace("\\n"," ")}{"..." if len(h["chunk"])>280 else ""}')
        ans = rag_answer(q, hits, args.chat_model, args.ollama_url)
        print("\n--- Answer ---")
        print(ans)

if __name__ == "__main__":
    main()
