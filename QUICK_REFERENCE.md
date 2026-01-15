# 🚀 SOLAT Quick Reference Card

## Installation (1 minute)
```bash
pip install -r requirements.txt
python3 TEST_RAG_SETUP.py
```

## Running (2 commands in different terminals)
```bash
# Terminal 1 - Backend
python3 run_sentinel.py

# Terminal 2 - Frontend
python3 run_dashboard.py
```

## Dashboard Navigation
- **🏠 Mission Control**: Regime status + signals
- **🔬 Market Analyzer**: Charts + surveillance
- **🧠 The Brain HMM**: Regime visualization
- **🧠 The Brain RAG**: PDF search
- **⚙️ Settings**: Configuration

## Using Local RAG
1. Copy PDFs to `data/knowledge_base/`
2. Click "🔄 Rebuild Index"
3. Type question in search box
4. Click "🚀 Search"

## File Structure
```
SOLAT_Platform/
├── src/
│   ├── core/
│   │   ├── engine.py (Sentinel backend)
│   │   ├── regime.py (HMM detector) ✨ NEW
│   │   └── ichimoku.py (Strategy)
│   └── knowledge/
│       └── brain.py (RAG system) ✨ NEW
├── dashboard/
│   ├── app.py (Main UI redesigned) ✨
│   └── pages/
│       └── brain_rag.py (RAG chat) ✨ NEW
├── data/
│   ├── knowledge_base/ (Your PDFs) ✨ NEW
│   ├── cache/brain/ (Embeddings) ✨ NEW
│   └── db/ (Database)
├── requirements.txt ✨ UPDATED
└── .gitignore ✨ NEW
```

## Key Commands
```python
# HMM Regime Detection
from src.core.regime import MarketRegimeDetector
detector = MarketRegimeDetector()
regime = detector.predict_regime(df)  # 'bull', 'bear', 'chop'

# Local RAG
from src.knowledge.brain import LocalKnowledgeBrain
brain = LocalKnowledgeBrain()
docs = brain.retrieve("What is momentum?", k=3)
docs, context = brain.answer_question("What is Ichimoku?", k=3)
```

## Performance Targets
| Operation | Time |
|-----------|------|
| Regime detection | 10ms |
| RAG search | <1ms |
| Full dashboard load | <1s |

## System Requirements
- Python 3.8+
- 2GB RAM minimum
- 500MB disk
- No GPU needed

## Troubleshooting
| Issue | Solution |
|-------|----------|
| No PDFs found | Add to `data/knowledge_base/` |
| No modules | `pip install -r requirements.txt` |
| Slow first load | Model downloads 80MB (automatic) |
| Database locked | Delete `data/db/` and restart |

## Documentation
- **Quick Start**: `BRAIN_QUICKSTART.md`
- **Dashboard**: `QUICK_START_DASHBOARD.md`
- **HMM Details**: `HMM_REGIME_UPGRADE.md`
- **RAG Details**: `RAG_LOCAL_BRAIN.md`
- **Status**: `PROJECT_STATUS_JANUARY_2026.md`

## Next Steps
1. Install dependencies
2. Add your PDFs
3. Start dashboard
4. Explore features
5. Monitor market regimes

---

**Status**: ✅ Production Ready
**Last Updated**: January 15, 2026
