import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="📰 News Summarizer", layout="centered")

@st.cache_resource
def get_pipe(model_name: str):
    return pipeline("summarization", model=model_name)

st.title("📰 News Summarizer")
model = st.selectbox("Model", ["facebook/bart-large-cnn", "google/pegasus-xsum"])
text = st.text_area("Paste a news article:", height=300)

if st.button("Summarize"):
    if len(text.strip()) == 0:
        st.warning("Please paste some text.")
    else:
        pipe = get_pipe(model)
        with st.spinner("Summarizing..."):
            out = pipe(text, truncation=True, max_length=128, min_length=48, do_sample=False)[0]
        st.subheader("Summary")
        st.write(out["summary_text"])
