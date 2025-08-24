import re
from bs4 import BeautifulSoup
from readability import Document

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")

def clean_html(html: str) -> str:
    if not html:
        return ""
    try:
        doc = Document(html)
        summary_html = doc.summary(html_partial=True)
        text = BeautifulSoup(summary_html, "lxml").get_text(" ")
    except Exception:
        text = BeautifulSoup(html, "lxml").get_text(" ")
    text = HTML_TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text

def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text or "").strip()
