#!/usr/bin/env python3
"""
Quick test script to verify RAG system is properly installed.
"""

import sys
import importlib

def check_import(module_name: str) -> bool:
    """Check if module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: {e}")
        return False

def main():
    print("🧠 SOLAT Local RAG - Setup Verification\n")
    
    print("Checking dependencies...")
    deps = [
        "pypdf",
        "sentence_transformers",
        "faiss",
        "numpy",
        "streamlit",
    ]
    
    all_ok = True
    for dep in deps:
        if not check_import(dep):
            all_ok = False
    
    print("\nChecking SOLAT modules...")
    try:
        from src.knowledge.brain import LocalKnowledgeBrain
        print("✅ src.knowledge.brain")
    except ImportError as e:
        print(f"❌ src.knowledge.brain: {e}")
        all_ok = False
    
    print("\nChecking directories...")
    from pathlib import Path
    
    dirs = [
        "data/knowledge_base",
        "data/cache/brain",
    ]
    
    for d in dirs:
        p = Path(d)
        if p.exists():
            print(f"✅ {d}")
        else:
            print(f"❌ {d} (creating...)")
            p.mkdir(parents=True, exist_ok=True)
            print(f"✅ {d} (created)")
    
    print("\n" + "="*50)
    if all_ok:
        print("🎉 All checks passed! RAG system ready to use.")
        print("\nNext steps:")
        print("1. Add PDFs to data/knowledge_base/")
        print("2. Run: python3 run_dashboard.py")
        print("3. Go to 🧠 The Brain page")
        print("4. Click 🔄 Rebuild Index")
        print("5. Start searching!")
        return 0
    else:
        print("⚠️  Some checks failed. Please install dependencies:")
        print("pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
