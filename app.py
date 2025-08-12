import os, json, hashlib, streamlit as st
from openai import OpenAI
from engine import Retriever, analyze
from prompts import SYSTEM_PROMPT, OPINION_PROMPT, CRITIQUE_PROMPT

# --- Auto-load secrets so you don't paste keys every run ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "REASONING_MODEL" in st.secrets:
    os.environ["REASONING_MODEL"] = st.secrets["REASONING_MODEL"]

st.set_page_config(page_title="Gerard Reasoning Engine", layout="wide")
st.title("Gerard Reasoning Engine — MVP")
st.caption("Grounded opinions + adversarial critique using your canon cards.")

# ---------- Modes ----------
MODES = {
    "Personal": {
        "top_k": 10,
        "prompt_opinion": OPINION_PROMPT,
        "prompt_critique": CRITIQUE_PROMPT,
        "pack_bias": {
            # pack id prefix -> multiplier
            "02_self_knowledge_awareness": 1.2,
            "03_emotional_education": 1.2,
            "01_integral_buddhism": 1.15,
            "04_romantic_realism": 1.1,
            "11_startup_canon": 0.9,
        },
    },
    "Work/Strategy": {
        "top_k": 13,
        "prompt_opinion": OPINION_PROMPT,
        "prompt_critique": CRITIQUE_PROMPT,
        "pack_bias": {
            "11_startup_canon": 1.25,
            "18_sensemaking_cynefin": 1.2,
            "13_systems_cybernetics": 1.15,
            "17_antifragility_decision_making": 1.15,
            "15_virtue_ethics": 1.05,
        },
    },
    "News": {
        "top_k": 12,
        "prompt_opinion": OPINION_PROMPT,
        "prompt_critique": CRITIQUE_PROMPT,
        "pack_bias": {
            # keep canon central; rely on pasted article as extra_docs
            "20_narrative_meaning": 1.15,
            "10_feminism": 1.1,
            "13_systems_cybernetics": 1.05,
        },
    },
    "Learning": {
        "top_k": 15,
        "prompt_opinion": OPINION_PROMPT,
        "prompt_critique": CRITIQUE_PROMPT,
        "pack_bias": {
            # encourage breadth
            "15_virtue_ethics": 1.1,
            "12_critical_rationalism": 1.1,
            "19_process_philosophy": 1.1,
            "14_phenomenology_enactivism": 1.1,
            "18_sensemaking_cynefin": 1.1,
        },
    },
    "Integral": {
    "top_k": 16,
    "prompt_opinion": OPINION_PROMPT,
    "prompt_critique": CRITIQUE_PROMPT,
    "pack_bias": {},  # no extra weighting: use the canon as-is
},
}

# ---------- Helpers ----------
def file_md5(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except FileNotFoundError:
        return ""

@st.cache_resource(show_spinner=False)
def get_retriever(canon_path: str, model: str, api_key: str):
    # cache key is (canon hash, model name, api key fingerprint)
    _hash = file_md5(canon_path)
    _finger = (api_key[:6] + "..." + api_key[-4:]) if api_key else "no-key"
    # construct and return retriever (Streamlit caches based on inputs)
    return Retriever(canon_path)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("OpenAI API Key", type="password",
                            value=os.getenv("OPENAI_API_KEY", ""), placeholder="sk-...")
    model = st.text_input("Reasoning model", value=os.getenv("REASONING_MODEL", "gpt-4o-mini"))
    canon_path = st.text_input("Canon JSONL path", value="data/canon_cards_enriched.jsonl")
    mode_name = st.selectbox("Mode", list(MODES.keys()), index=1)  # default Work/Strategy
    paste_mode = st.checkbox("I will paste article text", value=True)

    if st.button("Test OpenAI connection"):
        try:
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            if model:
                os.environ["REASONING_MODEL"] = model
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            client.chat.completions.create(
                model=os.environ.get("REASONING_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            st.success("OpenAI call succeeded.")
        except Exception as e:
            st.error(f"OpenAI error: {e}")

# ---------- Auto-init retriever (no manual 'Load canon') ----------
# Only initialize when we have a key + a canon file path
retriever = None
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
if model:
    os.environ["REASONING_MODEL"] = model
if api_key and canon_path and file_md5(canon_path):
    try:
        retriever = get_retriever(canon_path, os.environ.get("REASONING_MODEL",""), api_key)
    except Exception as e:
        st.sidebar.error(f"Failed to load canon: {e}")

# ---------- Main UI ----------
st.subheader("Ask a question or paste a link/topic")
query = st.text_area("Question / prompt",
                     placeholder="e.g., Is X policy likely to help with Y? What are the trade-offs?")

pasted = ""
if paste_mode:
    pasted = st.text_area("Optional: paste article or notes", height=200,
                          placeholder="Paste relevant text here…")

col1, col2 = st.columns([1, 2])
run = col1.button("Analyze")
show_ctx = col2.checkbox("Show evidence context", value=False)

if run:
    if retriever is None:
        st.error("Canon not loaded (missing key or file). Check sidebar.")
    elif not query.strip():
        st.error("Enter a question or topic.")
    else:
        cfg = MODES[mode_name]
        # Optional: adjust prompts per mode later; for now reuse base ones
        top_k = cfg["top_k"]

        with st.spinner(f"Analyzing ({mode_name})..."):
            try:
                opinion, critique, context = analyze(
                    query=query,
                    retriever=retriever,
                    system_prompt=SYSTEM_PROMPT,
                    opinion_prompt=cfg["prompt_opinion"],
                    critique_prompt=cfg["prompt_critique"],
                    pasted_text=pasted if pasted else None,
                )
                st.markdown("### Opinion")
                st.write(opinion)
                st.markdown("### Critique")
                st.write(critique)
                if show_ctx:
                    st.markdown("### Evidence Context")
                    st.code(context)
                ledger = {"mode": mode_name, "query": query, "opinion": opinion,
                          "critique": critique, "context": context}
                st.download_button(
                    "Download evidence ledger (JSON)",
                    data=json.dumps(ledger, indent=2),
                    file_name="evidence_ledger.json",
                    mime="application/json",
                )
            except Exception as e:
                st.error(f"Error: {e}")
