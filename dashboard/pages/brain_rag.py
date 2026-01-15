"""
🧠 The Brain - Local RAG Chat Interface

Query your personal PDF library with a natural language search interface.
100% local, no API calls, completely offline-capable.
"""

import logging
from pathlib import Path

import streamlit as st

from src.knowledge.brain import LocalKnowledgeBrain

logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="The Brain - RAG Chat",
    page_icon="🧠",
    layout="wide"
)

# ============================================================================
# SESSION STATE & INITIALIZATION
# ============================================================================


@st.cache_resource
def get_knowledge_brain():
    """Initialize knowledge brain with caching."""
    try:
        brain = LocalKnowledgeBrain(
            pdf_directory="data/knowledge_base",
            cache_dir="data/cache/brain",
            embedding_model="all-MiniLM-L6-v2"
        )
        return brain
    except Exception as e:
        logger.error(f"Failed to initialize knowledge brain: {e}")
        st.error(f"Error initializing knowledge brain: {e}")
        return None


# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

st.sidebar.title("🧠 Knowledge Base Controls")

# Check if data/knowledge_base exists
kb_path = Path("data/knowledge_base")
kb_path.mkdir(parents=True, exist_ok=True)

# Get PDF files
pdf_files = list(kb_path.glob("*.pdf"))

col1, col2 = st.sidebar.columns(2)

with col1:
    rebuild_pressed = st.button(
        "🔄 Rebuild Index",
        key="rebuild_knowledge",
        help="Re-index all PDFs in data/knowledge_base"
    )

with col2:
    refresh_pressed = st.button(
        "🔍 Refresh",
        key="refresh_knowledge",
        help="Refresh the file list"
    )

if rebuild_pressed:
    # Clear cache to force reinitialize
    if 'get_knowledge_brain' in st.cache_resource._cached_funcs:
        st.cache_resource.clear()
    st.success("✅ Knowledge index rebuilt. Refresh page to reload.")

st.sidebar.divider()

st.sidebar.subheader("📁 Available PDFs")

if pdf_files:
    st.sidebar.info(f"Found {len(pdf_files)} PDFs")
    for pdf_file in sorted(pdf_files):
        st.sidebar.caption(f"📄 {pdf_file.name}")
else:
    st.sidebar.warning(
        "No PDFs found. Please add PDF files to:\n\n"
        "`data/knowledge_base/`"
    )

st.sidebar.divider()

brain = get_knowledge_brain()

if brain:
    stats = brain.get_stats()
    st.sidebar.subheader("📊 Index Stats")
    st.sidebar.metric("Documents Indexed", stats["total_documents"])
    st.sidebar.metric("Embedding Model", "MiniLM-L6-v2")
    st.sidebar.caption(f"Cache: {stats['cache_directory']}")

# ============================================================================
# MAIN INTERFACE
# ============================================================================

st.title("🧠 The Brain - Your Personal Quant Library")

st.markdown("""
Ask questions about your trading PDFs. The system searches through your entire
library and returns the most relevant passages.

**How it works:**
1. Add PDF files to `data/knowledge_base/`
2. Click "Rebuild Index" to process them
3. Ask natural language questions
4. Get relevant excerpts from your library
""")

st.divider()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============================================================================
# SEARCH INTERFACE
# ============================================================================

st.subheader("🔍 Search Your Library")

col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input(
        "Ask your library a question...",
        placeholder="e.g., 'What is the formula for Ichimoku Cloud?'",
        key="rag_query"
    )

with col2:
    search_pressed = st.button("🚀 Search", key="search_button")

# ============================================================================
# SEARCH RESULTS
# ============================================================================

if search_pressed and query and brain:
    # Show search status
    with st.spinner("Searching knowledge base..."):
        docs, context = brain.answer_question(query, k=3)

    if docs:
        st.success(f"✅ Found {len(docs)} relevant document(s)")

        st.subheader("📚 Best Answers from Your Library")

        # Display results in tabs
        result_tabs = st.tabs([f"Result {i + 1}" for i in range(len(docs))])

        for idx, (tab, doc) in enumerate(zip(result_tabs, docs)):
            with tab:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Relevance", f"{doc['relevance']:.1%}")

                with col2:
                    st.metric("Source", doc["source"])

                with col3:
                    st.metric("Chunk ID", doc["chunk_id"])

                st.divider()

                st.markdown("**Relevant Excerpt:**")
                st.markdown(f"> {doc['text']}")

                st.caption(f"Characters: {doc['char_count']}")

        # Add to chat history
        st.session_state.chat_history.append({
            "query": query,
            "docs_found": len(docs),
            "top_source": docs[0]["source"] if docs else "None"
        })

    else:
        st.warning("❌ No relevant documents found. Try a different query.")

# ============================================================================
# CHAT HISTORY
# ============================================================================

if st.session_state.chat_history:
    st.divider()
    st.subheader("📜 Search History")

    with st.expander("Show recent searches", expanded=False):
        for idx, item in enumerate(reversed(st.session_state.chat_history[-10:]), 1):
            st.caption(
                f"{idx}. **{item['query'][:50]}...** "
                f"({item['docs_found']} docs, "
                f"Top: {item['top_source']})"
            )

# ============================================================================
# HELP SECTION
# ============================================================================

st.divider()

with st.expander("❓ How to Use The Brain", expanded=False):
    st.markdown("""
    ### Setup
    1. Create or use the `data/knowledge_base/` folder
    2. Add your PDF files (trading books, strategy docs, research papers)
    3. Click "Rebuild Index" to process them

    ### Searching
    - Ask natural language questions
    - The system finds relevant passages in your PDFs
    - Results show source document and relevance score

    ### Tips
    - Use specific trading terms (e.g., "Ichimoku", "volatility", "momentum")
    - Ask about formulas, definitions, strategy concepts
    - Combine multiple search terms for complex queries

    ### Technical Details
    - **Embedding Model**: all-MiniLM-L6-v2 (lightweight, fast)
    - **Search Algorithm**: FAISS vector similarity (L2 distance)
    - **Chunk Size**: 500 characters with 100 character overlap
    - **Results**: Top 3 most relevant passages per query
    - **Local Only**: No external APIs, 100% offline capable
    """)
