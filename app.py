import os, json, hashlib, time, datetime, streamlit as st
from openai import OpenAI
from engine import Retriever, analyze
from prompts import SYSTEM_PROMPT, OPINION_PROMPT, CRITIQUE_PROMPT

# --- Auto-load secrets ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "REASONING_MODEL" in st.secrets:
    os.environ["REASONING_MODEL"] = st.secrets["REASONING_MODEL"]

st.set_page_config(page_title="Gerard Reasoning Engine — MVP", layout="wide")
st.title("Gerard Reasoning Engine")
st.caption("Grounded opinions + adversarial critique using your canon cards, with persistent history.")

# ---------- Modes ----------
MODES = {
    "Personal": {"top_k": 10, "prompt_opinion": OPINION_PROMPT, "prompt_critique": CRITIQUE_PROMPT,
                 "pack_bias": {"02_self_knowledge_awareness": 1.2, "03_emotional_education": 1.2,
                               "01_integral_buddhism": 1.15, "04_romantic_realism": 1.1,
                               "11_startup_canon": 0.9}},
    "Work/Strategy": {"top_k": 13, "prompt_opinion": OPINION_PROMPT, "prompt_critique": CRITIQUE_PROMPT,
                      "pack_bias": {"11_startup_canon": 1.25, "18_sensemaking_cynefin": 1.2,
                                    "13_systems_cybernetics": 1.15, "17_antifragility_decision_making": 1.15,
                                    "15_virtue_ethics": 1.05}},
    "News": {"top_k": 12, "prompt_opinion": OPINION_PROMPT, "prompt_critique": CRITIQUE_PROMPT,
             "pack_bias": {"20_narrative_meaning": 1.15, "10_feminism": 1.1, "13_systems_cybernetics": 1.05}},
    "Learning": {"top_k": 15, "prompt_opinion": OPINION_PROMPT, "prompt_critique": CRITIQUE_PROMPT,
                 "pack_bias": {"15_virtue_ethics": 1.1, "12_critical_rationalism": 1.1,
                               "19_process_philosophy": 1.1, "14_phenomenology_enactivism": 1.1,
                               "18_sensemaking_cynefin": 1.1}},
    "Integral": {"top_k": 16, "prompt_opinion": OPINION_PROMPT, "prompt_critique": CRITIQUE_PROMPT, "pack_bias": {}},
}

# ---------- Helpers ----------
def file_md5(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except FileNotFoundError:
        return ""

@st.cache_resource(show_spinner=False)
def get_retriever(canon_path: str):
    return Retriever(canon_path)

def ensure_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []  # list of records
    if "active_id" not in st.session_state:
        st.session_state["active_id"] = None

ensure_state()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("OpenAI API Key", type="password",
                            value=os.getenv("OPENAI_API_KEY", ""), placeholder="sk-...")
    model = st.text_input("Reasoning model", value=os.getenv("REASONING_MODEL", "gpt-4o-mini"))
    canon_path = st.text_input("Canon JSONL path", value="data/canon_cards_enriched.jsonl")
    mode_name = st.selectbox("Mode", list(MODES.keys()), index=1)
    paste_mode = st.checkbox("Paste article/notes", value=True)

    colA, colB = st.columns(2)
    with colA:
        if st.button("Test OpenAI"):
            try:
                if api_key: os.environ["OPENAI_API_KEY"] = api_key
                if model: os.environ["REASONING_MODEL"] = model
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                client.chat.completions.create(
                    model=os.environ.get("REASONING_MODEL", "gpt-4o-mini"),
                    messages=[{"role":"user","content":"ping"}],
                    max_tokens=5,
                )
                st.success("OpenAI call OK.")
            except Exception as e:
                st.error(f"OpenAI error: {e}")
    with colB:
        if st.button("Clear history"):
            st.session_state["history"] = []
            st.session_state["active_id"] = None
            st.success("History cleared.")

# Prepare retriever (don’t block UI if canon missing)
retriever = None
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
if model:
    os.environ["REASONING_MODEL"] = model
if api_key and canon_path and file_md5(canon_path):
    try:
        retriever = get_retriever(canon_path)
    except Exception as e:
        st.sidebar.error(f"Failed to load canon: {e}")

# ---------- Input form (prevents reruns on each keystroke) ----------
with st.form("query_form", clear_on_submit=False):
    st.subheader("Ask a question or paste a link/topic")
    query = st.text_area("Question / prompt", key="query_input",
                         placeholder="e.g., Is X policy likely to help with Y? What are the trade-offs?")
    pasted = ""
    if paste_mode:
        pasted = st.text_area("Optional: paste article or notes", key="pasted_input", height=200)
    submitted = st.form_submit_button("Analyze", type="primary")

# ---------- Run analysis on submit and persist to history ----------
if submitted:
    if retriever is None:
        st.error("Canon not loaded (missing key or file). Check sidebar.")
    elif not (query and query.strip()):
        st.error("Enter a question or topic.")
    else:
        cfg = MODES[mode_name]
        with st.spinner(f"Analyzing ({mode_name})..."):
            try:
                opinion, critique, context = analyze(
                    query=query,
                    retriever=retriever,
                    system_prompt=SYSTEM_PROMPT,
                    opinion_prompt=cfg["prompt_opinion"],
                    critique_prompt=cfg["prompt_critique"],
                    pasted_text=pasted if pasted else None,
                    top_k=cfg["top_k"],
                    pack_bias=cfg["pack_bias"],
                )
                # Persist result
                ts = time.time()
                rec_id = f"{int(ts)}-{len(st.session_state['history'])+1}"
                record = {
                    "id": rec_id,
                    "timestamp": ts,
                    "timestamp_iso": datetime.datetime.utcfromtimestamp(ts).isoformat() + "Z",
                    "mode": mode_name,
                    "query": query,
                    "pasted_preview": (pasted[:200] + ("…" if pasted and len(pasted) > 200 else "")) if pasted else "",
                    "opinion": opinion,
                    "critique": critique,
                    "context": context,
                }
                st.session_state["history"].insert(0, record)  # most recent first
                st.session_state["active_id"] = rec_id
                st.success("Analysis added to history.")
            except Exception as e:
                st.error(f"Error: {e}")

# ---------- History panel ----------
st.markdown("### History")
hist = st.session_state["history"]
if hist:
    options = [f"{i+1}. [{r['mode']}] {r['timestamp_iso']} — {r['query'][:60]}{'…' if len(r['query'])>60 else ''}" for i, r in enumerate(hist)]
    idx_map = {options[i]: hist[i]["id"] for i in range(len(hist))}
    default_idx = 0
    default_label = options[default_idx]
    sel_label = st.selectbox("Select a prior analysis", options, index=0 if st.session_state["active_id"] is None else
                             next((i for i,o in enumerate(options) if idx_map[o]==st.session_state["active_id"]), 0))
    active_id = idx_map[sel_label]
    st.session_state["active_id"] = active_id

    # Export buttons
    col1, col2 = st.columns([1,1])
    with col1:
        if st.download_button("Download selected (JSON)",
                              data=json.dumps(next(r for r in hist if r["id"]==active_id), indent=2),
                              file_name="analysis.json", mime="application/json"):
            pass
    with col2:
        if st.download_button("Download all history (JSON)",
                              data=json.dumps(hist, indent=2),
                              file_name="analysis_history.json", mime="application/json"):
            pass

    # Render active record
    rec = next(r for r in hist if r["id"] == active_id)
    st.markdown(f"#### {rec['mode']} — {rec['timestamp_iso']}")
    st.markdown("**Query**")
    st.write(rec["query"])
    if rec["pasted_preview"]:
        st.markdown("**Pasted preview**")
        st.code(rec["pasted_preview"])
    st.markdown("**Opinion**")
    st.write(rec["opinion"])
    st.markdown("**Critique**")
    st.write(rec["critique"])
    if st.checkbox("Show evidence context", key="show_ctx", value=False):
        st.code(rec["context"])
else:
    st.info("No history yet. Run an analysis using the form above.")
