"""
RAG Mind AI — v5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Full Light + Dark theme  (auto-detect + manual toggle, no rerun)
✅ Voice input in chat row
✅ Hybrid Search (BM25 + Semantic + RRF)
✅ Auto Summarization
✅ TTS Voice Output
✅ Export (PDF + Word)
✅ Eval & Scoring

Free stack: Groq LLM · HuggingFace Embeddings · ChromaDB
"""

import os, shutil, tempfile
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

st.set_page_config(
    page_title="RAG Mind AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.loaders       import load_file, load_web, chunk_documents
from utils.vectorstore   import build_vectorstore, add_documents_to_vectorstore
from utils.rag_chain     import query_rag, GROQ_MODELS, format_context, build_rag_chain
from utils.hybrid_search import HybridRetriever
from utils.summarizer    import summarize_document, multi_doc_overview
from utils.voice         import text_to_speech, get_audio_html, transcribe_audio_file
from utils.exporter      import export_to_pdf, export_to_docx
from utils.evaluator     import evaluate_response, score_label

# ══════════════════════════════════════════════════════════════════
#  THEME SYSTEM
#  Strategy:
#   • CSS custom properties split into [data-theme="light"] and
#     [data-theme="dark"] on <html> — zero Python reruns needed.
#   • A tiny JS snippet reads localStorage on load and listens for
#     the toggle button click, swapping data-theme instantly.
#   • @media (prefers-color-scheme) sets the initial default so
#     the page always matches the OS before JS runs (no flash).
# ══════════════════════════════════════════════════════════════════
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

/* ── LIGHT palette ── */
:root, [data-theme="light"] {
    --bg-base:        #f4f6fb;
    --bg-surface:     #ffffff;
    --bg-card:        #ffffff;
    --bg-hover:       #eef2ff;
    --border:         #e2e8f0;
    --border-bright:  #c7d2fe;

    --accent:         #4f46e5;
    --accent-glow:    rgba(79,70,229,0.18);
    --accent-bright:  #4338ca;
    --accent-soft:    #ede9fe;
    --amber:          #d97706;
    --amber-glow:     rgba(217,119,6,0.12);
    --green:          #059669;
    --green-glow:     rgba(5,150,105,0.10);
    --red:            #e11d48;

    --text-primary:   #0f172a;
    --text-secondary: #475569;
    --text-muted:     #94a3b8;

    --shadow-sm:  0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md:  0 4px 12px rgba(0,0,0,0.08);
    --shadow-glow:0 0 24px rgba(79,70,229,0.12);

    --hero-grad1: rgba(79,70,229,0.10);
    --hero-grad2: rgba(217,119,6,0.07);
    --hero-grad3: rgba(5,150,105,0.05);

    --scrollbar-thumb: #c7d2fe;
    --chat-user-bg:   #ede9fe;
    --chat-ai-bg:     #f8fafc;
}

/* ── DARK palette ── */
[data-theme="dark"] {
    --bg-base:        #0d0f1a;
    --bg-surface:     #12162b;
    --bg-card:        #181d35;
    --bg-hover:       #1e2540;
    --border:         rgba(255,255,255,0.07);
    --border-bright:  rgba(255,255,255,0.15);

    --accent:         #6366f1;
    --accent-glow:    rgba(99,102,241,0.28);
    --accent-bright:  #818cf8;
    --accent-soft:    rgba(99,102,241,0.15);
    --amber:          #f59e0b;
    --amber-glow:     rgba(245,158,11,0.20);
    --green:          #10b981;
    --green-glow:     rgba(16,185,129,0.15);
    --red:            #f43f5e;

    --text-primary:   #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted:     #475569;

    --shadow-sm:  0 1px 3px rgba(0,0,0,0.4);
    --shadow-md:  0 4px 16px rgba(0,0,0,0.5);
    --shadow-glow:0 0 30px rgba(99,102,241,0.15);

    --hero-grad1: rgba(99,102,241,0.18);
    --hero-grad2: rgba(245,158,11,0.10);
    --hero-grad3: rgba(16,185,129,0.08);

    --scrollbar-thumb: #1e2540;
    --chat-user-bg:   rgba(99,102,241,0.15);
    --chat-ai-bg:     #181d35;
}

/* ── OS default before JS runs ── */
@media (prefers-color-scheme: light) {
    :root { /* already light — no-op */ }
}
@media (prefers-color-scheme: dark) {
    :root {
        --bg-base:        #0d0f1a;
        --bg-surface:     #12162b;
        --bg-card:        #181d35;
        --bg-hover:       #1e2540;
        --border:         rgba(255,255,255,0.07);
        --border-bright:  rgba(255,255,255,0.15);
        --accent:         #6366f1;
        --accent-glow:    rgba(99,102,241,0.28);
        --accent-bright:  #818cf8;
        --accent-soft:    rgba(99,102,241,0.15);
        --amber:          #f59e0b;
        --amber-glow:     rgba(245,158,11,0.20);
        --green:          #10b981;
        --green-glow:     rgba(16,185,129,0.15);
        --red:            #f43f5e;
        --text-primary:   #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted:     #475569;
        --shadow-sm:  0 1px 3px rgba(0,0,0,0.4);
        --shadow-md:  0 4px 16px rgba(0,0,0,0.5);
        --shadow-glow:0 0 30px rgba(99,102,241,0.15);
        --hero-grad1: rgba(99,102,241,0.18);
        --hero-grad2: rgba(245,158,11,0.10);
        --hero-grad3: rgba(16,185,129,0.08);
        --scrollbar-thumb: #1e2540;
        --chat-user-bg:   rgba(99,102,241,0.15);
        --chat-ai-bg:     #181d35;
    }
}

/* ── Shared radii ── */
:root {
    --radius-sm: 6px;
    --radius-md: 12px;
    --radius-lg: 20px;
    --radius-xl: 28px;
}

/* ═══════════════════════════════════
   APP SHELL
═══════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
section[data-testid="stMain"] > div {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: background-color 0.25s ease, color 0.25s ease;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
    transition: background 0.25s ease;
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

[data-testid="stHeader"] { background: transparent !important; }

/* Scrollbars */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: var(--scrollbar-thumb);
    border-radius: 99px;
}

/* Typography */
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; }

/* ═══════════════════════════════════
   FORM INPUTS
═══════════════════════════════════ */
.stTextInput input,
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: var(--bg-card) !important;
    border-color: var(--border-bright) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: all 0.2s ease;
}
.stTextInput input::placeholder { color: var(--text-muted) !important; }
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
div[data-baseweb="select"] span { color: var(--text-primary) !important; }

/* ═══════════════════════════════════
   BUTTONS
═══════════════════════════════════ */
.stButton > button {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    box-shadow: var(--shadow-sm) !important;
}
.stButton > button:hover {
    background: var(--bg-hover) !important;
    border-color: var(--accent) !important;
    transform: translateY(-1px);
    box-shadow: var(--shadow-md) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent) 0%, #4338ca 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px var(--accent-glow) !important;
}
.stButton > button[kind="primary"]:hover {
    filter: brightness(1.08);
    box-shadow: 0 6px 20px var(--accent-glow) !important;
    transform: translateY(-1px);
}

/* ═══════════════════════════════════
   TABS
═══════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-surface) !important;
    border-radius: var(--radius-md) !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm);
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
    transition: all 0.15s ease;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px var(--accent-glow);
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

/* ═══════════════════════════════════
   EXPANDERS
═══════════════════════════════════ */
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

/* ═══════════════════════════════════
   SLIDERS & CHECKBOXES
═══════════════════════════════════ */
[data-testid="stSlider"] > div > div > div > div { background: var(--accent) !important; }
[data-testid="stCheckbox"] span { color: var(--text-primary) !important; }

/* ═══════════════════════════════════
   FILE UPLOADER
═══════════════════════════════════ */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border-bright) !important;
    border-radius: var(--radius-md) !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }
[data-testid="stFileUploader"] * { color: var(--text-secondary) !important; }

/* ═══════════════════════════════════
   CHAT
═══════════════════════════════════ */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 4px 0 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
}

/* Chat input bar */
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1.5px solid var(--border-bright) !important;
    border-radius: var(--radius-xl) !important;
    box-shadow: var(--shadow-glow) !important;
    transition: all 0.2s ease;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow), var(--shadow-md) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Audio input */
[data-testid="stAudioInput"] {
    background: var(--accent-soft) !important;
    border: 1.5px solid var(--accent) !important;
    border-radius: var(--radius-lg) !important;
    padding: 0.5rem 1rem !important;
    box-shadow: 0 0 20px var(--accent-glow) !important;
    transition: all 0.2s;
}

/* ═══════════════════════════════════
   METRICS
═══════════════════════════════════ */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem 1.2rem !important;
    box-shadow: var(--shadow-sm);
    transition: all 0.2s;
}
[data-testid="stMetric"]:hover { box-shadow: var(--shadow-md); transform: translateY(-1px); }
[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--accent-bright) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.5rem !important;
}

/* ═══════════════════════════════════
   ALERTS / NOTIFICATIONS
═══════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border) !important;
    font-family: 'DM Sans', sans-serif !important;
    background: var(--bg-card) !important;
}

/* Progress bar */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent-bright)) !important;
    border-radius: 99px !important;
}

/* ═══════════════════════════════════
   THEME TOGGLE BUTTON  (floating)
═══════════════════════════════════ */
#theme-toggle-btn {
    position: fixed;
    top: 14px;
    right: 70px;
    z-index: 99999;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    border: 1.5px solid var(--border-bright);
    background: var(--bg-card);
    color: var(--text-primary);
    font-size: 17px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: var(--shadow-md);
    transition: all 0.22s cubic-bezier(.4,0,.2,1);
    outline: none;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}
#theme-toggle-btn:hover {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
    transform: rotate(20deg) scale(1.1);
    box-shadow: 0 0 18px var(--accent-glow);
}
#theme-toggle-btn .icon-sun  { display: none; }
#theme-toggle-btn .icon-moon { display: inline; }
[data-theme="light"] #theme-toggle-btn .icon-sun  { display: inline; }
[data-theme="light"] #theme-toggle-btn .icon-moon { display: none; }

/* ═══════════════════════════════════
   CUSTOM COMPONENTS
═══════════════════════════════════ */

/* Hero */
.hero-wrap {
    position: relative;
    padding: 2.5rem 0 1.5rem;
    margin-bottom: 1rem;
    overflow: hidden;
    border-radius: var(--radius-lg);
}
.hero-bg {
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 60% 80% at 15% 50%, var(--hero-grad1) 0%, transparent 70%),
        radial-gradient(ellipse 40% 60% at 85% 20%, var(--hero-grad2) 0%, transparent 70%),
        radial-gradient(ellipse 50% 50% at 60% 80%, var(--hero-grad3) 0%, transparent 70%);
    pointer-events: none;
    transition: background 0.3s ease;
}
.hero-eyebrow {
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--accent-bright);
    margin-bottom: 0.5rem;
    display: flex; align-items: center; gap: 8px;
}
.hero-eyebrow::before {
    content: ''; display: inline-block;
    width: 24px; height: 2px;
    background: var(--accent-bright); border-radius: 99px;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 800;
    line-height: 1.05;
    color: var(--text-primary);
    letter-spacing: -0.03em;
}
.hero-title .grad {
    background: linear-gradient(135deg, var(--accent-bright) 0%, var(--amber) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    margin-top: 0.7rem;
    font-size: 1rem; color: var(--text-secondary);
    font-weight: 300; letter-spacing: 0.01em;
}
.hero-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 1.2rem; }
.pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 12px; border-radius: 99px;
    font-size: 0.74rem; font-weight: 600; letter-spacing: 0.03em;
    border: 1px solid; transition: transform 0.15s;
}
.pill:hover { transform: translateY(-1px); }
.pill-indigo { background: rgba(99,102,241,0.12); color: var(--accent-bright); border-color: rgba(99,102,241,0.3); }
.pill-amber  { background: rgba(245,158,11,0.10); color: var(--amber);         border-color: rgba(245,158,11,0.3); }
.pill-green  { background: rgba(16,185,129,0.10); color: var(--green);         border-color: rgba(16,185,129,0.3); }
.pill-red    { background: rgba(244,63,94,0.10);  color: var(--red);           border-color: rgba(244,63,94,0.3); }

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
    transition: border-left-color 0.2s, box-shadow 0.2s;
    box-shadow: var(--shadow-sm);
}
.src-card:hover { border-left-color: var(--amber); box-shadow: var(--shadow-md); }
.src-card b { color: var(--text-primary); font-weight: 600; font-size: 0.84rem; display: block; margin-bottom: 3px; }

/* Eval card */
.eval-wrap {
    background: linear-gradient(135deg, var(--green-glow), transparent);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: var(--radius-md);
    padding: 1rem 1.2rem;
    margin-top: 0.75rem;
    transition: background 0.25s;
}
.eval-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem; font-weight: 700;
    color: var(--green); margin-bottom: 0.6rem;
}
.eval-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 16px; }
.eval-row {
    display: flex; justify-content: space-between;
    font-size: 0.8rem; color: var(--text-secondary); padding: 2px 0;
}
.eval-row span:last-child { color: var(--green); font-weight: 600; }

/* Stat chip */
.stat-chip {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 0.8rem 1rem; text-align: center;
    box-shadow: var(--shadow-sm);
    transition: all 0.2s;
}
.stat-chip:hover { box-shadow: var(--shadow-md); transform: translateY(-1px); }
.stat-chip .num {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem; font-weight: 800;
    color: var(--accent-bright); line-height: 1;
}
.stat-chip .lbl {
    font-size: 0.7rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.1em; margin-top: 3px;
}

/* Groq tip */
.groq-tip {
    background: var(--accent-soft);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: var(--radius-sm);
    padding: 0.65rem 0.9rem;
    font-size: 0.82rem;
    color: var(--accent-bright);
    margin-bottom: 0.5rem;
    transition: background 0.25s;
}
.groq-tip a { color: var(--amber) !important; text-decoration: none; font-weight: 600; }
.groq-tip a:hover { text-decoration: underline; }

/* Section header */
.sec-hdr {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem; font-weight: 700;
    color: var(--text-primary);
    margin: 0.5rem 0 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
}

/* Sidebar label */
.sb-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.13em; text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.2rem 0 0.4rem;
}

/* Voice banner */
.voice-heard {
    background: var(--accent-soft);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: var(--radius-md);
    padding: 0.6rem 1rem;
    font-size: 0.85rem;
    color: var(--accent-bright);
    margin-bottom: 0.5rem;
    display: flex; align-items: center; gap: 8px;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    background: var(--bg-card);
    border: 1.5px dashed var(--border-bright);
    border-radius: var(--radius-lg);
    color: var(--text-secondary);
    box-shadow: var(--shadow-sm);
    transition: background 0.25s;
}
.empty-state .icon { font-size: 2.5rem; margin-bottom: 0.8rem; }
.empty-state h3 {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem; color: var(--text-primary); margin-bottom: 0.4rem;
}
.empty-state ol {
    text-align: left; display: inline-block;
    margin-top: 0.6rem; line-height: 2.2; font-size: 0.9rem;
}
.empty-state a { color: var(--accent-bright); font-weight: 500; }
</style>
"""

THEME_JS = """
<script>
(function () {
    const HTML  = document.documentElement;
    const STORE = 'ragmind-theme';

    /* determine initial theme */
    function getInitial() {
        const saved = localStorage.getItem(STORE);
        if (saved) return saved;
        return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }

    function applyTheme(t) {
        HTML.setAttribute('data-theme', t);
        localStorage.setItem(STORE, t);
        const btn = document.getElementById('theme-toggle-btn');
        if (btn) btn.title = t === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
    }

    function toggle() {
        const cur = HTML.getAttribute('data-theme') || 'light';
        applyTheme(cur === 'dark' ? 'light' : 'dark');
    }

    /* inject button */
    function injectButton() {
        if (document.getElementById('theme-toggle-btn')) return;
        const btn = document.createElement('button');
        btn.id = 'theme-toggle-btn';
        btn.innerHTML = '<span class="icon-sun">☀️</span><span class="icon-moon">🌙</span>';
        btn.setAttribute('aria-label', 'Toggle theme');
        btn.addEventListener('click', toggle);
        document.body.appendChild(btn);
    }

    /* apply immediately to avoid flash */
    applyTheme(getInitial());

    /* inject button when DOM ready */
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', injectButton);
    } else {
        injectButton();
    }

    /* Streamlit re-renders — keep re-injecting the button */
    const obs = new MutationObserver(() => {
        injectButton();
        /* re-apply theme attr in case Streamlit wiped it */
        const saved = localStorage.getItem(STORE);
        if (saved && HTML.getAttribute('data-theme') !== saved) applyTheme(saved);
    });
    obs.observe(document.body, { childList: true, subtree: true });

    /* listen to OS theme changes */
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
        if (!localStorage.getItem(STORE)) applyTheme(e.matches ? 'dark' : 'light');
    });
})();
</script>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown(THEME_JS,  unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
DEFAULTS = dict(
    vectorstore=None, all_chunks=[], chat_history=[],
    doc_count=0, chunk_count=0, summaries={},
    doc_sources=[], hybrid_retriever=None, eval_history=[],
)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("✦ **RAG Mind AI**")
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
    sel_label    = st.selectbox("Model", model_labels)
    model        = model_keys[model_labels.index(sel_label)]

    st.markdown('<div class="sb-label">Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload", type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True, label_visibility="collapsed",
    )
    web_url = st.text_input("Web URL", placeholder="https://...", label_visibility="collapsed")

    st.markdown('<div class="sb-label">Options</div>', unsafe_allow_html=True)
    use_hybrid = st.checkbox("⚡ Hybrid Search",  value=True, help="BM25 + semantic + RRF fusion")
    show_eval  = st.checkbox("📊 Eval scores",    value=True)
    voice_out  = st.checkbox("🔊 Voice output",   value=False)

    with st.expander("⚙️ Advanced"):
        chunk_size    = st.slider("Chunk size",    200, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk overlap",   0,  500,  200,  50)
        top_k         = st.slider("Top-k chunks",    1,   10,    5)
        temperature   = st.slider("Temperature",   0.0,  1.0,  0.0, 0.1)

    ingest_btn = st.button("🚀 Ingest Documents", use_container_width=True, type="primary")

    st.markdown("---")
    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        if st.session_state.vectorstore:
            try: st.session_state.vectorstore.delete_collection()
            except: pass
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
                    try: all_docs.extend(load_web(web_url.strip()))
                    except Exception as e: st.error(f"URL: {e}")
                if all_docs:
                    chunks = chunk_documents(all_docs, chunk_size, chunk_overlap)

            if all_docs:
                with st.spinner(f"Embedding {len(chunks)} chunks…"):
                    try:
                        vs = (build_vectorstore(chunks) if st.session_state.vectorstore is None
                              else add_documents_to_vectorstore(st.session_state.vectorstore, chunks))
                        st.session_state.vectorstore  = vs
                        st.session_state.all_chunks  += chunks
                        st.session_state.doc_count   += len({d.metadata.get("source","") for d in all_docs})
                        st.session_state.chunk_count += len(chunks)
                        if use_hybrid:
                            st.session_state.hybrid_retriever = HybridRetriever(
                                st.session_state.all_chunks, vs, k=top_k)
                        st.session_state.doc_sources += list(
                            {d.metadata.get("source","") for d in all_docs if d.metadata.get("source","")})
                        st.success(f"✅ {len(all_docs)} pages → {len(chunks)} chunks!")
                    except Exception as e:
                        st.error(f"Embed error: {e}")

# ══════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
  <div class="hero-bg"></div>
  <div class="hero-title">RAG Mind <span class="grad">AI</span></div>
  <div class="hero-sub">Query your documents with intelligence. Grounded answers, zero hallucinations.</div>
  <div class="hero-pills">
    <span class="pill pill-indigo">⚡ Hybrid Search</span>
    <span class="pill pill-amber">✦ Auto Summarize</span>
    <span class="pill pill-green">🎙 Voice I/O</span>
    <span class="pill pill-red">📊 Eval Scoring</span>
    <span class="pill pill-indigo">📥 Export PDF/Word</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab_chat, tab_summary, tab_export, tab_eval = st.tabs([
    "💬  Chat", "📋  Summaries", "📥  Export", "📊  Evaluation"
])

# ── helpers ───────────────────────────────────────────────────────
def render_sources(source_docs):
    with st.expander("📎 Sources used", expanded=False):
        for src in source_docs:
            source   = src.metadata.get("source", "Unknown")
            page     = src.metadata.get("page", "")
            page_str = f" · p.{page+1}" if page != "" else ""
            snippet  = src.page_content[:260].replace("\n", " ")
            st.markdown(
                f'<div class="src-card"><b>{source}{page_str}</b>{snippet}…</div>',
                unsafe_allow_html=True)

def render_eval(ev):
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

# ╔══════════════════╗
# ║  TAB 1 — CHAT    ║
# ╚══════════════════╝
with tab_chat:
    # History
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn.get("sources"):   render_sources(turn["sources"])
            if show_eval and turn.get("eval"): render_eval(turn["eval"])

    # Voice widget (only when KB ready)
    voice_question = None
    audio_input_fn = getattr(st, "audio_input", getattr(st, "experimental_audio_input", None))
    if audio_input_fn and st.session_state.vectorstore:
        col_v, _ = st.columns([3, 1])
        with col_v:
            audio_val = audio_input_fn("🎙️ Speak your question — or type below", key="voice_in")
            if audio_val:
                with st.spinner("Transcribing…"):
                    voice_question = transcribe_audio_file(audio_val.getvalue(), file_format="wav")
                if voice_question:
                    st.markdown(
                        f'<div class="voice-heard">🎤 Heard: <b>"{voice_question}"</b></div>',
                        unsafe_allow_html=True)
                else:
                    st.warning("Couldn't transcribe. Try again or type below.")

    # Chat input
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
                placeholder, full_response = st.empty(), ""
                try:
                    if use_hybrid and st.session_state.hybrid_retriever:
                        source_docs = st.session_state.hybrid_retriever.search(question)
                        stream = build_rag_chain(api_key, model, temperature).stream(
                            {"context": format_context(source_docs), "question": question})
                    else:
                        stream, source_docs = query_rag(
                            st.session_state.vectorstore, question, api_key,
                            model=model, k=top_k, temperature=temperature)

                    for chunk in stream:
                        full_response += chunk.content if hasattr(chunk, "content") else str(chunk)
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)

                    if source_docs: render_sources(source_docs)

                    ev = evaluate_response(question, full_response, source_docs)
                    if show_eval: render_eval(ev)

                    if voice_out:
                        ab = text_to_speech(full_response[:1500])
                        if ab: st.markdown(get_audio_html(ab), unsafe_allow_html=True)

                    st.session_state.chat_history.append(
                        {"question": question, "answer": full_response,
                         "sources": source_docs, "eval": ev})
                    st.session_state.eval_history.append(ev)

                except Exception as e:
                    st.error(f"Error: {e}")

    if not st.session_state.vectorstore:
        st.markdown("""
        <div class="empty-state">
          <div class="icon">✦</div>
          <h3>Your knowledge base is empty</h3>
          <p>Upload documents or paste a URL in the sidebar to get started.</p>
          <ol>
            <li>Get a free Groq key at <a href="https://console.groq.com" target="_blank">console.groq.com</a></li>
            <li>Upload PDF, DOCX, TXT, or MD files (or paste a URL)</li>
            <li>Click <b>Ingest Documents</b></li>
            <li>Ask questions — or speak them 🎙️</li>
          </ol>
        </div>""", unsafe_allow_html=True)

# ╔═════════════════════════╗
# ║  TAB 2 — SUMMARIES      ║
# ╚═════════════════════════╝
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
                summaries, prog, total = {}, st.progress(0), len(source_chunks)
                for i, (src, chks) in enumerate(source_chunks.items()):
                    with st.spinner(f"Summarizing {src}…"):
                        try: summaries[src] = summarize_document(chks, api_key, src, model)
                        except Exception as e: st.error(f"{src}: {e}")
                    prog.progress((i+1)/total)
                st.session_state.summaries = summaries
                if len(summaries) > 1:
                    with st.spinner("Building multi-doc overview…"):
                        try:
                            ov = multi_doc_overview(list(summaries.values()), api_key, model)
                            st.session_state.summaries["__overview__"] = {"summary": ov, "source": "All Documents"}
                        except Exception as e: st.warning(f"Overview failed: {e}")
                prog.empty()
                st.success("✅ Summaries generated!")

        if st.session_state.summaries:
            if "__overview__" in st.session_state.summaries:
                with st.expander("🌐 Multi-Document Overview", expanded=True):
                    st.write(st.session_state.summaries["__overview__"]["summary"])
            for src, s in st.session_state.summaries.items():
                if src == "__overview__": continue
                with st.expander(f"📄 {s.get('source', src)}", expanded=False):
                    if s.get("doc_type"): st.caption(f"Type: {s['doc_type']}")
                    st.markdown("**Summary**");   st.write(s.get("summary", ""))
                    if s.get("key_topics"):
                        st.markdown("**Key Topics**"); st.write(s.get("key_topics", ""))

# ╔═══════════════════╗
# ║  TAB 3 — EXPORT   ║
# ╚═══════════════════╝
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
        export_title = st.text_input("Document title", value="RAG Mind Q&A Session")
        col_pdf, col_docx = st.columns(2)
        with col_pdf:
            if st.button("📄 Generate PDF", use_container_width=True, type="primary"):
                with st.spinner("Building PDF…"):
                    try:
                        pdf = export_to_pdf(st.session_state.chat_history,
                                            st.session_state.doc_sources, export_title)
                        st.download_button("⬇️ Download PDF", data=pdf,
                            file_name="ragmind_session.pdf", mime="application/pdf",
                            use_container_width=True)
                    except Exception as e: st.error(f"PDF error: {e}")
        with col_docx:
            if st.button("📝 Generate Word", use_container_width=True, type="primary"):
                with st.spinner("Building Word doc…"):
                    try:
                        docx = export_to_docx(st.session_state.chat_history,
                                              st.session_state.doc_sources, export_title)
                        st.download_button("⬇️ Download Word", data=docx,
                            file_name="ragmind_session.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True)
                    except Exception as e: st.error(f"Word error: {e}")

# ╔═════════════════════════════╗
# ║  TAB 4 — EVALUATION         ║
# ╚═════════════════════════════╝
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
        avg = {k: sum(e[k] for e in evs)/len(evs) for k in evs[0]}
        st.markdown(f"#### Session Quality: {score_label(avg['overall'])}")
        for col, (label, key) in zip(st.columns(5), [
            ("🎯 Overall",        "overall"),
            ("🔒 Faithfulness",   "faithfulness"),
            ("🔍 Context Rel.",   "context_relevance"),
            ("💬 Answer Rel.",    "answer_relevance"),
            ("📝 ROUGE-L",        "rouge_l"),
        ]):
            with col: st.metric(label, f"{avg[key]:.0%}")
        st.markdown("---")
        st.markdown("#### Per-Question Breakdown")
        for i, (turn, ev) in enumerate(zip(st.session_state.chat_history, evs), 1):
            with st.expander(f"Q{i}: {turn['question'][:75]}…", expanded=False):
                for col, (k, lbl) in zip(st.columns(4), [
                    ("faithfulness","Faithfulness"),("context_relevance","Context Rel."),
                    ("answer_relevance","Answer Rel."),("rouge_l","ROUGE-L"),
                ]):
                    with col: st.metric(lbl, f"{ev[k]:.0%}")
                st.caption(f"Overall: {score_label(ev['overall'])}")
