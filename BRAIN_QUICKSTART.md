# 🧠 The Brain - Quick Start Guide

**Status**: ✅ Ready to use
**Setup Time**: 5 minutes
**No external APIs required**: 100% local

---

## 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `pypdf` - Read PDFs
- `sentence-transformers` - Generate embeddings
- `faiss-cpu` - Vector search
- `numpy` - Numerical operations

**Note**: First time will download the embedding model (~80MB). This is automatic.

---

## 2️⃣ Add Your PDFs

```bash
# Copy PDFs to this folder:
data/knowledge_base/

# Example:
cp ~/Downloads/trading_book.pdf data/knowledge_base/
cp ~/Documents/ichimoku_guide.pdf data/knowledge_base/
```

**File Types**: Any PDF with extractable text

---

## 3️⃣ Launch Dashboard

**Terminal 1: Start Sentinel (optional)**
```bash
python3 run_sentinel.py
```

**Terminal 2: Start Dashboard**
```bash
python3 run_dashboard.py
```

Browser opens: http://localhost:8501

---

## 4️⃣ Navigate to The Brain

In dashboard sidebar, select: **🧠 The Brain**

---

## 5️⃣ Rebuild Knowledge Index

Click button: **🔄 Rebuild Index**

**Output**:
```
✓ Loaded ichimoku_guide.pdf: 45 chunks
✓ Loaded trading_book.pdf: 120 chunks
✓ Generated 165 embeddings
✓ Created and cached knowledge index
```

---

## 6️⃣ Search Your PDFs

Type in search box:
```
"What is the Ichimoku Cloud?"
"How do I use momentum?"
"What is support and resistance?"
```

Click: **🚀 Search**

---

## 7️⃣ View Results

Results show:
- **Relevance Score** (0-100%)
- **Source Document** (PDF filename)
- **Relevant Excerpt** (exact passage from PDF)

---

## That's It! 🎉

You now have a personal trading library inside your terminal.

---

## Common Workflows

### Add a New PDF

1. Copy PDF to `data/knowledge_base/`
2. Click "Rebuild Index" in sidebar
3. Start searching

### Search Best Practices

| Good Questions | Bad Questions |
|---|---|
| "What is Ichimoku Cloud?" | "cloud" |
| "How to calculate momentum?" | "calc" |
| "Risk management strategies" | "risk" |
| "Support and resistance levels" | "support" |

**Tip**: Use complete phrases for best results.

### Export Results

Copy/paste relevant excerpts directly from results to:
- Trading notebook
- Strategy document
- Research file

---

## Troubleshooting

### "No PDFs found"

1. Check folder exists: `data/knowledge_base/`
2. Add PDF files to folder
3. Click "Rebuild Index"

### "No relevant documents found"

Try rephrasing:
- ❌ "momentum"
- ✅ "How do I measure momentum?"

### "Slow first load"

First time downloads 80MB model. Later loads are instant.

---

## What's Happening Behind the Scenes

```
Your PDF Library
     ↓
Extract Text (pypdf)
     ↓
Split into Chunks (500 chars, 100 char overlap)
     ↓
Generate Embeddings (sentence-transformers)
     ↓
Build Vector Index (FAISS)
     ↓
Cache on Disk (data/cache/brain/)
     ↓
Ready for Search ⚡
```

When you search:
```
Your Question
     ↓
Embed Question (sentence-transformers)
     ↓
Search Index (FAISS - <1ms)
     ↓
Return Top 3 Passages
     ↓
Display with Source & Relevance
```

---

## Technical Details

| Component | What It Does |
|-----------|--------------|
| **pypdf** | Reads PDFs, extracts text |
| **sentence-transformers** | Converts text to 384-dimensional vectors |
| **FAISS** | Searches vectors super fast |
| **Cache** | Stores embeddings so you don't regenerate |

**Result**: Search 1000+ pages in <100ms with zero external APIs

---

## Next Steps

1. ✅ Add your favorite trading books
2. ✅ Build your personal trading library
3. ✅ Search while trading
4. ✅ Never lose a trading insight again

---

## Questions?

See `RAG_LOCAL_BRAIN.md` for full documentation.

