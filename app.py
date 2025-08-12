import os, json, streamlit as st
from engine import Retriever, analyze
from prompts import SYSTEM_PROMPT, OPINION_PROMPT, CRITIQUE_PROMPT
from openai import OpenAI

# --- Auto-load Streamlit Cloud secrets (so you don't have to paste the key every run) ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "REASONING_MODEL" in st.secrets:
    os.environ["REASONING_MODEL"] = st.secrets["REASONING_MODEL"]
# ----------------------------------------------------------------------------------------

st.set_page_config(page_title="Gerard Reasoning Engine", layout="wide")

st.title("Gerard Reasoning Engine — MVP")
st.caption("Grounded opinions + adversarial critique using your canon cards.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", value=os.getenv("OPENAI_API_KEY", ""))
    model = st.text_input("Reasoning model (optional)", value=os.getenv("REASONING_MODEL", "gpt-4o-mini"))
    canon_path = st.text_input("Canon JSONL path", value="data/canon_cards_enriched.jsonl")

    # Test button to verify key/model access quickly
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

    if st.button("Load canon"):
        try:
            # ensure env picks up sidebar values if provided
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            if model:
                os.environ["REASONING_MODEL"] = model
            st.session_state["retriever"] = Retriever(canon_path)
            st.success("Canon loaded.")
        except Exception as e:
            st.error(f"Error loading canon: {e}")

    st.markdown("---")
    st.header("Options")
    paste_mode = st.checkbox("I will paste article text", value=True)
    st.markdown("---")
    st.caption("Tip: keep pasted text to a few thousand words.")

st.subheader("Ask a question or paste a link/topic")
query = st.text_area("Question / prompt", placeholder="e.g., Is X policy likely to help with Y? What are the trade-offs?")
pasted = ""
if paste_mode:
    pasted = st.text_area("Optional: paste article or notes", height=200, placeholder="Paste relevant text here…")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    run = st.button("Analyze")
with col2:
    clear = st.button("Clear")
with col3:
    show_ctx = st.checkbox("Show evidence context", value=False)

if clear:
    st.session_state.pop("retriever", None)
    st.experimental_rerun()

if run:
    if "retriever" not in st.session_state:
        st.error("Load canon first from the sidebar.")
    elif not query.strip():
        st.error("Enter a question or topic.")
    else:
        retriever = st.session_state["retriever"]
        with st.spinner("Thinking..."):
            try:
                opinion, critique, context = analyze(
                    query,
                    retriever,
                    SYSTEM_PROMPT,
                    OPINION_PROMPT,
                    CRITIQUE_PROMPT,
                    pasted_text=pasted if pasted else None,
                )
                st.markdown("### Opinion")
                st.write(opinion)
                st.markdown("### Critique")
                st.write(critique)
                if show_ctx:
                    st.markdown("### Evidence Context")
                    st.code(context)
                ledger = {"query": query, "opinion": opinion, "critique": critique, "context": context}
                st.download_button(
                    "Download evidence ledger (JSON)",
                    data=json.dumps(ledger, indent=2),
                    file_name="evidence_ledger.json",
                    mime="application/json",
                )
            except Exception as e:
                st.error(f"Error: {e}")
