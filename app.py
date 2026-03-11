"""
RAG Application v4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ No Login / Auth (removed)
✅ Premium Dark UI  (custom fonts, mesh gradients, glassmorphism)
✅ Voice input integrated directly into chat input row
✅ Hybrid Search (BM25 + Semantic + RRF)
✅ Auto Summarization (per-doc + multi-doc overview)
✅ TTS Voice Output (gTTS)
✅ Export (PDF + Word)
✅ Eval & Scoring (faithfulness, relevance, ROUGE-L)

Free stack: Groq LLM · HuggingFace Embeddings · ChromaDB
"""

import os, shutil, tempfile
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

st.set_page_config(
    page_title=" RAG Mind AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ────────────────────────────────────────────────────────────────────
from utils.loaders       import load_file, load_web, chunk_documents
from utils.vectorstore   import build_vectorstore, add_documents_to_vectorstore
from utils.rag_chain     import query_rag, GROQ_MODELS, format_context, build_rag_chain
from utils.hybrid_search import HybridRetriever
from utils.summarizer    import summarize_document, multi_doc_overview
from utils.voice         import text_to_speech, get_audio_html, transcribe_audio_file
from utils.exporter      import export_to_pdf, export_to_docx
from utils.evaluator     import evaluate_response, score_label

# ══════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Dark editorial theme
# Fonts: "Syne" (display) + "DM Sans" (body) — distinctive, modern
# Colors: deep navy bg, electric indigo accent, warm amber highlight
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg-base:       #0d0f1a;
    --bg-surface:    #12162b;
    --bg-card:       #181d35;
    --bg-hover:      #1e2540;
    --border:        rgba(255,255,255,0.07);
    --border-bright: rgba(255,255,255,0.14);

    --accent:        #6366f1;
    --accent-glow:   rgba(99,102,241,0.25);
    --accent-bright: #818cf8;
    --amber:         #f59e0b;
    --amber-glow:    rgba(245,158,11,0.2);
    --green:         #10b981;
    --green-glow:    rgba(16,185,129,0.15);
    --red:           #f43f5e;

    --text-primary:   #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted:     #475569;

    --radius-sm:  6px;
    --radius-md:  12px;
    --radius-lg:  20px;
    --radius-xl:  28px;
}

/* ── App shell ── */
html, body, [data-testid="stAppViewContainer"], .main {
    background-color: var(--bg-base) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stHeader"] { background: transparent !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--bg-hover); border-radius: 99px; }

/* ── Typography ── */
h1,h2,h3,h4 { font-family: 'Syne', sans-serif !important; }

/* ── Streamlit overrides ── */
.stTextInput input, .stSelectbox select,
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: var(--bg-card) !important;
    border-color: var(--border-bright) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput input:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--accent-glow) !important; }

.stButton > button {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: var(--bg-hover) !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent) 0%, #4f46e5 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px var(--accent-glow) !important;
}
.stButton > button[kind="primary"]:hover {
    filter: brightness(1.1);
    box-shadow: 0 6px 20px var(--accent-glow) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-surface) !important;
    border-radius: var(--radius-md) !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
}
.streamlit-expanderContent {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div > div {
    background: var(--accent) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border-bright) !important;
    border-radius: var(--radius-md) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 4px 0 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.65 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: var(--radius-xl) !important;
    box-shadow: 0 0 30px rgba(99,102,241,0.08) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Audio input widget ── */
[data-testid="stAudioInput"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--accent) !important;
    border-radius: var(--radius-lg) !important;
    padding: 0.5rem 1rem !important;
    box-shadow: 0 0 20px var(--accent-glow) !important;
}

/* ── Metric ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetric"] label { color: var(--text-secondary) !important; font-size: 0.8rem !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--accent-bright) !important; font-family: 'Syne', sans-serif !important; }

/* ── Info / Success / Error boxes ── */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ══════════════════════════════════════════
   CUSTOM COMPONENT CLASSES
══════════════════════════════════════════ */

/* Hero header */
.hero-wrap {
    position: relative;
    padding: 2.5rem 0 1.5rem;
    margin-bottom: 1rem;
    overflow: hidden;
}
.hero-bg {
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 60% 80% at 15% 50%, rgba(99,102,241,0.18) 0%, transparent 70%),
        radial-gradient(ellipse 40% 60% at 85% 20%, rgba(245,158,11,0.10) 0%, transparent 70%),
        radial-gradient(ellipse 50% 50% at 60% 80%, rgba(16,185,129,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent-bright);
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 24px; height: 2px;
    background: var(--accent-bright);
    border-radius: 99px;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 800;
    line-height: 1.05;
    color: var(--text-primary);
    letter-spacing: -0.03em;
}
.hero-title span {
    background: linear-gradient(135deg, var(--accent-bright) 0%, var(--amber) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    margin-top: 0.8rem;
    font-size: 1rem;
    color: var(--text-secondary);
    font-weight: 300;
    letter-spacing: 0.01em;
}
.hero-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 1.2rem;
}
.pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    border: 1px solid;
}
.pill-indigo { background: rgba(99,102,241,0.12); color: var(--accent-bright); border-color: rgba(99,102,241,0.3); }
.pill-amber  { background: rgba(245,158,11,0.10); color: var(--amber); border-color: rgba(245,158,11,0.3); }
.pill-green  { background: rgba(16,185,129,0.10); color: var(--green); border-color: rgba(16,185,129,0.3); }
.pill-red    { background: rgba(244,63,94,0.10);  color: var(--red);   border-color: rgba(244,63,94,0.3); }

/* Source card */
.src-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    padding: 0.8rem 1.1rem;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    margin-bottom: 0.5rem;
    font-size: 0.83rem;
    color: var(--text-secondary);
    transition: border-color 0.2s;
}
.src-card:hover { border-left-color: var(--amber); }
.src-card b { color: var(--text-primary); font-weight: 600; font-size: 0.84rem; display: block; margin-bottom: 3px; }

/* Eval card */
.eval-wrap {
    background: linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(16,185,129,0.03) 100%);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: var(--radius-md);
    padding: 1rem 1.2rem;
    margin-top: 0.75rem;
}
.eval-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--green);
    margin-bottom: 0.6rem;
}
.eval-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px 16px;
}
.eval-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: var(--text-secondary);
    padding: 2px 0;
}
.eval-row span:last-child { color: var(--green); font-weight: 600; }

/* Sidebar stat chip */
.stat-chip {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 0.8rem 1rem;
    text-align: center;
}
.stat-chip .num {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--accent-bright);
    line-height: 1;
}
.stat-chip .lbl {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 3px;
}

/* Groq tip box */
.groq-tip {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: var(--radius-sm);
    padding: 0.7rem 0.9rem;
    font-size: 0.82rem;
    color: var(--accent-bright);
    margin-bottom: 0.5rem;
}
.groq-tip a { color: var(--amber) !important; text-decoration: none; font-weight: 600; }
.groq-tip a:hover { text-decoration: underline; }

/* Section header */
.sec-hdr {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0.5rem 0 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    background: var(--bg-card);
    border: 1px dashed var(--border-bright);
    border-radius: var(--radius-lg);
    color: var(--text-secondary);
}
.empty-state .icon { font-size: 2.5rem; margin-bottom: 0.8rem; }
.empty-state h3 {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    color: var(--text-primary);
    margin-bottom: 0.4rem;
}
.empty-state ol {
    text-align: left;
    display: inline-block;
    margin-top: 0.6rem;
    line-height: 2;
    font-size: 0.9rem;
}
.empty-state a { color: var(--accent-bright); }

/* Sidebar section label */
.sb-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.2rem 0 0.4rem;
}

/* Voice heard banner */
.voice-heard {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: var(--radius-md);
    padding: 0.6rem 1rem;
    font-size: 0.85rem;
    color: var(--accent-bright);
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
DEFAULTS = dict(
    vectorstore=None, all_chunks=[], chat_history=[],
    doc_count=0, chunk_count=0, summaries={},
    doc_sources=[], hybrid_retriever=None, eval_history=[],
    voice_transcript=None,
)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("✦ ** RAG Mind AI**", unsafe_allow_html=False)
    st.markdown('<div class="sb-label">API Configuration</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="groq-tip">🔑 Free key → <a href="https://console.groq.com" target="_blank">console.groq.com</a></div>',
        unsafe_allow_html=True,
    )

    api_key = st.text_input(
        "Groq API Key", type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        placeholder="gsk_...", label_visibility="collapsed",
    )

    model_labels = list(GROQ_MODELS.values())
    model_keys   = list(GROQ_MODELS.keys())
    sel_label    = st.selectbox("Model", model_labels, label_visibility="visible")
    model        = model_keys[model_labels.index(sel_label)]

    st.markdown('<div class="sb-label">Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload", type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True, label_visibility="collapsed",
    )
    web_url = st.text_input("Web URL", placeholder="https://...", label_visibility="collapsed")

    st.markdown('<div class="sb-label">Options</div>', unsafe_allow_html=True)
    use_hybrid = st.checkbox("⚡ Hybrid Search", value=True, help="BM25 + semantic + RRF fusion")
    show_eval  = st.checkbox("📊 Eval scores",   value=True)
    voice_out  = st.checkbox("🔊 Voice output",  value=False)

    with st.expander("⚙️ Advanced"):
        chunk_size    = st.slider("Chunk size",    200, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk overlap",   0,  500,  200,  50)
        top_k         = st.slider("Top-k chunks",    1,   10,    5)
        temperature   = st.slider("Temperature",   0.0,  1.0,  0.0, 0.1)

    ingest_btn = st.button("🚀 Ingest Documents", use_container_width=True, type="primary")

    st.markdown("---")
    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        if st.session_state.vectorstore:
            try:
                st.session_state.vectorstore.delete_collection()
            except Exception:
                pass
        if os.path.exists("./data/chroma_db"):
            shutil.rmtree("./data/chroma_db", ignore_errors=True)
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.success("Knowledge base cleared.")

    if st.session_state.vectorstore:
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="stat-chip"><div class="num">{st.session_state.doc_count}</div><div class="lbl">Docs</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-chip"><div class="num">{st.session_state.chunk_count}</div><div class="lbl">Chunks</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# INGEST
# ══════════════════════════════════════════════════════════════════
if ingest_btn:
    if not api_key:
        st.sidebar.error("Enter your Groq API key first.")
    elif not uploaded_files and not web_url.strip():
        st.sidebar.warning("Add at least one file or URL.")
    else:
        with st.sidebar:
            with st.spinner("Loading & chunking…"):
                all_docs = []
                for uf in uploaded_files:
                    suffix = Path(uf.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uf.read()); tmp_path = tmp.name
                    try:
                        docs = load_file(tmp_path)
                        for d in docs: d.metadata["source"] = uf.name
                        all_docs.extend(docs)
                    except Exception as e:
                        st.error(f"{uf.name}: {e}")
                    finally:
                        os.unlink(tmp_path)
                if web_url.strip():
                    try:
                        all_docs.extend(load_web(web_url.strip()))
                    except Exception as e:
                        st.error(f"URL: {e}")
                if all_docs:
                    chunks = chunk_documents(all_docs, chunk_size, chunk_overlap)

            if all_docs:
                with st.spinner(f"Embedding {len(chunks)} chunks…"):
                    try:
                        if st.session_state.vectorstore is None:
                            vs = build_vectorstore(chunks)
                        else:
                            vs = add_documents_to_vectorstore(st.session_state.vectorstore, chunks)
                        st.session_state.vectorstore  = vs
                        st.session_state.all_chunks  += chunks
                        st.session_state.doc_count   += len({d.metadata.get("source","") for d in all_docs})
                        st.session_state.chunk_count += len(chunks)
                        if use_hybrid:
                            st.session_state.hybrid_retriever = HybridRetriever(
                                st.session_state.all_chunks, vs, k=top_k
                            )
                        new_sources = list({d.metadata.get("source","") for d in all_docs if d.metadata.get("source","")})
                        st.session_state.doc_sources += new_sources
                        st.success(f"✅ {len(all_docs)} pages → {len(chunks)} chunks!")
                    except Exception as e:
                        st.error(f"Embed error: {e}")

# ══════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
  <div class="hero-bg"></div>
  <div class="hero-title">RAG Mind <span>AI</span></div>
  <div class="hero-sub">Query your documents with intelligence. Get grounded answers, not hallucinations.</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab_chat, tab_summary, tab_export, tab_eval = st.tabs([
    "💬  Chat", "📋  Summaries", "📥  Export", "📊  Evaluation"
])

# ╔══════════════════════════════════════╗
# ║  TAB 1 — CHAT                        ║
# ╚══════════════════════════════════════╝
with tab_chat:

    # ── Chat history ──────────────────────────────────────────────
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn.get("sources"):
                with st.expander("📎 Sources used", expanded=False):
                    for src in turn["sources"]:
                        source   = src.metadata.get("source", "Unknown")
                        page     = src.metadata.get("page", "")
                        page_str = f" · p.{page+1}" if page != "" else ""
                        snippet  = src.page_content[:260].replace("\n", " ")
                        st.markdown(
                            f'<div class="src-card"><b>{source}{page_str}</b>{snippet}…</div>',
                            unsafe_allow_html=True,
                        )
            if show_eval and turn.get("eval"):
                ev = turn["eval"]
                st.markdown(f"""
                <div class="eval-wrap">
                  <div class="eval-title">Answer Quality · {score_label(ev["overall"])}</div>
                  <div class="eval-grid">
                    <div class="eval-row"><span>Faithfulness</span><span>{ev["faithfulness"]:.0%}</span></div>
                    <div class="eval-row"><span>Context Rel.</span><span>{ev["context_relevance"]:.0%}</span></div>
                    <div class="eval-row"><span>Answer Rel.</span><span>{ev["answer_relevance"]:.0%}</span></div>
                    <div class="eval-row"><span>ROUGE-L</span><span>{ev["rouge_l"]:.0%}</span></div>
                  </div>
                </div>""", unsafe_allow_html=True)

    # ── Voice input — integrated above chat input ──────────────────
    audio_input_fn = getattr(st, "audio_input", getattr(st, "experimental_audio_input", None))
    voice_question = None

    if audio_input_fn and st.session_state.vectorstore:
        col_voice, col_spacer = st.columns([3, 1])
        with col_voice:
            audio_val = audio_input_fn("🎙️ Speak your question — or type below", key="voice_in")
            if audio_val:
                with st.spinner("Transcribing…"):
                    voice_question = transcribe_audio_file(audio_val.getvalue(), file_format="wav")
                if voice_question:
                    st.markdown(
                        f'<div class="voice-heard">🎤 Heard: <b>"{voice_question}"</b></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("Couldn't transcribe. Try again or type below.")

    # ── Text chat input ────────────────────────────────────────────
    question = st.chat_input(
        "Ask anything about your documents…" if st.session_state.vectorstore else "⬅ Ingest documents first",
        disabled=(st.session_state.vectorstore is None),
    ) or voice_question

    if question:
        if not api_key:
            st.error("Enter your Groq API key in the sidebar.")
        else:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                placeholder   = st.empty()
                full_response = ""
                try:
                    # Retrieval
                    if use_hybrid and st.session_state.hybrid_retriever:
                        source_docs = st.session_state.hybrid_retriever.search(question)
                        context     = format_context(source_docs)
                        chain       = build_rag_chain(api_key, model, temperature)
                        stream      = chain.stream({"context": context, "question": question})
                    else:
                        stream, source_docs = query_rag(
                            st.session_state.vectorstore, question, api_key,
                            model=model, k=top_k, temperature=temperature,
                        )

                    for chunk in stream:
                        token = chunk.content if hasattr(chunk, "content") else str(chunk)
                        full_response += token
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)

                    # Sources
                    if source_docs:
                        with st.expander("📎 Sources used", expanded=False):
                            for src in source_docs:
                                source   = src.metadata.get("source", "Unknown")
                                page     = src.metadata.get("page", "")
                                page_str = f" · p.{page+1}" if page != "" else ""
                                snippet  = src.page_content[:260].replace("\n", " ")
                                st.markdown(
                                    f'<div class="src-card"><b>{source}{page_str}</b>{snippet}…</div>',
                                    unsafe_allow_html=True,
                                )

                    # Eval
                    ev = evaluate_response(question, full_response, source_docs)
                    if show_eval:
                        st.markdown(f"""
                        <div class="eval-wrap">
                          <div class="eval-title">Answer Quality · {score_label(ev["overall"])}</div>
                          <div class="eval-grid">
                            <div class="eval-row"><span>Faithfulness</span><span>{ev["faithfulness"]:.0%}</span></div>
                            <div class="eval-row"><span>Context Rel.</span><span>{ev["context_relevance"]:.0%}</span></div>
                            <div class="eval-row"><span>Answer Rel.</span><span>{ev["answer_relevance"]:.0%}</span></div>
                            <div class="eval-row"><span>ROUGE-L</span><span>{ev["rouge_l"]:.0%}</span></div>
                          </div>
                        </div>""", unsafe_allow_html=True)

                    # TTS output
                    if voice_out:
                        audio_bytes = text_to_speech(full_response[:1500])
                        if audio_bytes:
                            st.markdown(get_audio_html(audio_bytes), unsafe_allow_html=True)

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer":   full_response,
                        "sources":  source_docs,
                        "eval":     ev,
                    })
                    st.session_state.eval_history.append(ev)

                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Empty state ────────────────────────────────────────────────
    if st.session_state.vectorstore is None:
        st.markdown("""
        <div class="empty-state">
          <div class="icon">✦</div>
          <h3>Your knowledge base is empty</h3>
          <p>Upload documents or paste a URL in the sidebar to get started.</p>
          <ol>
            <li>Get a free Groq key at <a href="https://console.groq.com" target="_blank">console.groq.com</a></li>
            <li>Upload PDF, DOCX, TXT, or MD files (or paste a URL)</li>
            <li>Click <b>Ingest Documents</b></li>
            <li>Ask questions or speak them 🎙️</li>
          </ol>
        </div>
        """, unsafe_allow_html=True)

# ╔══════════════════════════════════════╗
# ║  TAB 2 — SUMMARIES                   ║
# ╚══════════════════════════════════════╝
with tab_summary:
    st.markdown('<div class="sec-hdr">Auto Document Summarization</div>', unsafe_allow_html=True)

    if not st.session_state.vectorstore:
        st.info("Ingest documents first to generate summaries.")
    else:
        if st.button("✨ Generate Summaries", type="primary"):
            if not api_key:
                st.error("Groq API key required.")
            else:
                source_chunks = defaultdict(list)
                for chunk in st.session_state.all_chunks:
                    source_chunks[chunk.metadata.get("source", "Unknown")].append(chunk)

                summaries = {}
                prog = st.progress(0)
                total = len(source_chunks)
                for i, (src, chks) in enumerate(source_chunks.items()):
                    with st.spinner(f"Summarizing {src}…"):
                        try:
                            summaries[src] = summarize_document(chks, api_key, src, model)
                        except Exception as e:
                            st.error(f"{src}: {e}")
                    prog.progress((i + 1) / total)

                st.session_state.summaries = summaries

                if len(summaries) > 1:
                    with st.spinner("Building multi-doc overview…"):
                        try:
                            overview = multi_doc_overview(list(summaries.values()), api_key, model)
                            st.session_state.summaries["__overview__"] = {
                                "summary": overview, "source": "All Documents"
                            }
                        except Exception as e:
                            st.warning(f"Overview failed: {e}")
                prog.empty()
                st.success("✅ Summaries generated!")

        if st.session_state.summaries:
            if "__overview__" in st.session_state.summaries:
                with st.expander("🌐 Multi-Document Overview", expanded=True):
                    st.write(st.session_state.summaries["__overview__"]["summary"])

            for src, s in st.session_state.summaries.items():
                if src == "__overview__": continue
                with st.expander(f"📄 {s.get('source', src)}", expanded=False):
                    if s.get("doc_type"):
                        st.caption(f"Document type: {s['doc_type']}")
                    st.markdown("**Summary**")
                    st.write(s.get("summary", ""))
                    if s.get("key_topics"):
                        st.markdown("**Key Topics**")
                        st.write(s.get("key_topics", ""))

# ╔══════════════════════════════════════╗
# ║  TAB 3 — EXPORT                      ║
# ╚══════════════════════════════════════╝
with tab_export:
    st.markdown('<div class="sec-hdr">Export Q&A Session</div>', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""
        <div class="empty-state">
          <div class="icon">📥</div>
          <h3>Nothing to export yet</h3>
          <p>Ask some questions in the Chat tab first.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.write(f"**{len(st.session_state.chat_history)} Q&A pairs** ready to export.")
        export_title = st.text_input("Document title", value="DocMind Q&A Session")

        col_pdf, col_docx = st.columns(2)
        with col_pdf:
            if st.button("📄 Generate PDF", use_container_width=True, type="primary"):
                with st.spinner("Building PDF…"):
                    try:
                        pdf_bytes = export_to_pdf(
                            st.session_state.chat_history,
                            st.session_state.doc_sources,
                            export_title,
                        )
                        st.download_button(
                            "⬇️ Download PDF", data=pdf_bytes,
                            file_name="docmind_session.pdf", mime="application/pdf",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"PDF error: {e}")

        with col_docx:
            if st.button("📝 Generate Word", use_container_width=True, type="primary"):
                with st.spinner("Building Word doc…"):
                    try:
                        docx_bytes = export_to_docx(
                            st.session_state.chat_history,
                            st.session_state.doc_sources,
                            export_title,
                        )
                        st.download_button(
                            "⬇️ Download Word", data=docx_bytes,
                            file_name="docmind_session.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"Word error: {e}")

# ╔══════════════════════════════════════╗
# ║  TAB 4 — EVALUATION DASHBOARD        ║
# ╚══════════════════════════════════════╝
with tab_eval:
    st.markdown('<div class="sec-hdr">Answer Quality Dashboard</div>', unsafe_allow_html=True)

    if not st.session_state.eval_history:
        st.markdown("""
        <div class="empty-state">
          <div class="icon">📊</div>
          <h3>No evaluation data yet</h3>
          <p>Ask questions in the Chat tab to see quality scores here.</p>
        </div>""", unsafe_allow_html=True)
    else:
        evs = st.session_state.eval_history
        n   = len(evs)
        avg = {k: sum(e[k] for e in evs) / n for k in evs[0]}

        st.markdown(f"#### Session Quality: {score_label(avg['overall'])}")

        cols = st.columns(5)
        for col, (label, key) in zip(cols, [
            ("🎯 Overall",           "overall"),
            ("🔒 Faithfulness",      "faithfulness"),
            ("🔍 Context Rel.",      "context_relevance"),
            ("💬 Answer Rel.",       "answer_relevance"),
            ("📝 ROUGE-L",           "rouge_l"),
        ]):
            with col:
                st.metric(label, f"{avg[key]:.0%}")

        st.markdown("---")
        st.markdown("#### Per-Question Breakdown")

        for i, (turn, ev) in enumerate(zip(st.session_state.chat_history, evs), 1):
            with st.expander(f"Q{i}: {turn['question'][:75]}…", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                for col, (k, lbl) in zip([c1, c2, c3, c4], [
                    ("faithfulness",      "Faithfulness"),
                    ("context_relevance", "Context Rel."),
                    ("answer_relevance",  "Answer Rel."),
                    ("rouge_l",           "ROUGE-L"),
                ]):
                    with col:
                        st.metric(lbl, f"{ev[k]:.0%}")
                st.caption(f"Overall: {score_label(ev['overall'])}")
