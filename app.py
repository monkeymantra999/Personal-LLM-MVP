import os, json, hashlib, streamlit as st
from openai import OpenAI
from engine import Retriever, analyze
from prompts import SYSTEM_PROMPT, OPINION_PROMPT, CRITIQUE_PROMPT

# --- Auto-load secrets ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "REASONING_MODEL" in st.secrets:
    os.environ["REASONING_MODEL"] = st.secrets["REASONING_MODEL"]

st.set_page_config(page_title="Gerard Reasoning Engine", layout="wide")
st.title("Gerard Reasoning Engine — MVP")

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

def file_md5(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except FileNotFoundError:
        return ""

@st.cache_resource(show_spinner=False)
def get_retriever(canon_path: str):
    return Retriever(canon_path)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""), placeholder="sk-...")
    model = st.text_input("Reasoning model", value=os.getenv("REASONING_MODEL", "gpt-4o-mini"))
    canon_path = st.text_input("Canon JSONL path", value="data/canon_cards_enriched.jsonl")
    mode_name = st.selectbox("Mode", list(MODES.keys()), index=1)
    paste_mode = st.checkbox("I will paste article text", value=True)
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if model:
        os.environ["REASONING_MODEL"] = model

retriever = None
if api_key and canon_path and file_md5(canon_path):
    retriever = get_retriever(canon_path)

# ---------- Main ----------
st.subheader("Ask a question or paste a link/topic")
query = st.text_area("Question / prompt", placeholder="Type your query here...")
pasted = st.text_area("Optional: paste article or notes", height=200) if paste_mode else ""

if st.button("Analyze"):
    if retriever is None:
        st.error("Canon not loaded (missing key or file).")
    elif not query.strip():
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
                st.markdown("### Opinion")
                st.write(opinion)
                st.markdown("### Critique")
                st.write(critique)
                if st.checkbox("Show evidence context"):
                    st.code(context)
            except Exception as e:
                st.error(f"Error: {e}")
