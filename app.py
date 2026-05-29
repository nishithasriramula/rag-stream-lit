import re
import streamlit as st
from pypdf import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="PDF Q&A", layout="wide")
st.title("📚 PDF Q&A App")
st.markdown(
    "Use the controls below to upload a PDF, inspect page content, search for keywords, ask questions, or retrieve the nearest relevant text segments."
)

MODE_OPTIONS = ["Ask a question", "Summarise document"]
mode = st.selectbox("Select mode", MODE_OPTIONS)
uploaded_file = st.file_uploader("Upload PDF", type="pdf")


@st.cache_resource
def load_encoder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def build_chunks(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue

        if current_len + len(words) > chunk_size and current:
            chunks.append(" ".join(current).strip())
            overlap_words = current[-overlap:] if overlap else []
            current = overlap_words.copy()
            current_len = len(current)

        current.extend(words)
        current_len += len(words)

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


@st.cache_data
def embed_chunks(chunks: list[str]) -> np.ndarray:
    model = load_encoder()
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def nearest_chunks(
    query: str, chunks: list[str], embeddings: np.ndarray, top_k: int = 3
) -> list[tuple[float, str]]:
    if not query.strip() or len(chunks) == 0:
        return []

    model = load_encoder()
    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    scores = np.dot(embeddings, query_embedding)
    best_indices = np.argsort(-scores)[:top_k]
    return [(float(scores[i]), chunks[i]) for i in best_indices]


def summarize_text(text: str, max_sentences: int = 5) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return "No text available to summarise."

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= max_sentences:
        return text

    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "was",
        "were",
        "been",
        "have",
        "has",
        "had",
        "shall",
        "would",
        "could",
        "should",
        "which",
        "their",
        "there",
        "these",
        "those",
        "also",
        "among",
        "about",
        "other",
        "such",
        "only",
        "when",
    }

    words = re.findall(r"\w+", text.lower())
    freq: dict[str, int] = {}
    for word in words:
        if word in stopwords or len(word) < 3:
            continue
        freq[word] = freq.get(word, 0) + 1

    sentence_scores: list[tuple[int, str]] = []
    for sentence in sentences:
        score = sum(freq.get(word.lower(), 0) for word in re.findall(r"\w+", sentence))
        sentence_scores.append((score, sentence))

    top_sentences = sorted(sentence_scores, key=lambda x: x[0], reverse=True)[
        :max_sentences
    ]
    top_sentences = sorted(top_sentences, key=lambda x: sentences.index(x[1]))
    summary = " ".join(sentence for _, sentence in top_sentences)
    return summary or "Unable to generate a summary."


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("-\n", "").strip())


def search_text(text: str, query: str) -> list[tuple[int, int, str]]:
    if not query.strip():
        return []

    normalized_text = normalize_text(text)
    normalized_query = normalize_text(query)
    pattern = re.compile(re.escape(normalized_query), re.IGNORECASE)
    matches = []
    for match in pattern.finditer(normalized_text):
        start = max(0, match.start() - 80)
        end = min(len(normalized_text), match.end() + 80)
        snippet = normalized_text[start:end]
        matches.append((match.start(), match.end(), snippet))
    return matches


def metadata_text(metadata: dict | None) -> str:
    if not metadata:
        return "No metadata available."

    lines = []
    for key, value in metadata.items():
        if not value:
            continue
        clean_key = key.lstrip("/").replace("/", " ").title()
        lines.append(f"**{clean_key}:** {value}")

    return "\n\n".join(lines) if lines else "No metadata available."


if uploaded_file:
    try:
        reader = PdfReader(uploaded_file)
    except Exception as exc:
        st.error(f"Unable to read PDF: {exc}")
    else:
        page_count = len(reader.pages)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        if not text.strip():
            st.error("No text could be extracted from the PDF.")
        else:
            st.success("PDF uploaded successfully")
            chunks = build_chunks(text)
            embeddings = embed_chunks(chunks)
            metadata = getattr(reader, "metadata", None)

            info_col, preview_col = st.columns([1, 2])

            with info_col:
                st.subheader("Document info")
                st.write(f"**Pages:** {page_count}")
                st.markdown(metadata_text(metadata))
                st.divider()

                search_keyword = st.text_input(
                    "Search keywords in PDF",
                    placeholder="e.g. enrollment, address",
                )
                if search_keyword:
                    matches = search_text(text, search_keyword)
                    if matches:
                        st.success(f"Found {len(matches)} exact match(es)")
                        for i, (_, _, snippet) in enumerate(matches[:3], start=1):
                            st.markdown(f"**Match {i}:** {snippet}")
                    else:
                        st.warning("No exact matches found for that keyword.")

                nearest_query = st.text_input(
                    "Find nearest content for",
                    placeholder="Enter a query to retrieve the nearest text segments",
                )
                if nearest_query:
                    nearest = nearest_chunks(nearest_query, chunks, embeddings, top_k=4)
                    if nearest:
                        st.success("Nearest content retrieved")
                        for i, (score, snippet) in enumerate(nearest, start=1):
                            st.markdown(
                                f"**Nearest {i}** — similarity: {score:.3f}\n\n{snippet}"
                            )
                    else:
                        st.warning("Unable to retrieve nearest content.")

            with preview_col:
                st.subheader("Page preview")
                preview_page = st.slider("Select page", 1, page_count, 1)
                preview_text = (
                    reader.pages[preview_page - 1].extract_text()
                    or "No text found on this page."
                )
                st.text_area("Extracted page text", preview_text, height=260)

                show_raw = st.checkbox("Show full extracted text")
                if show_raw:
                    st.text_area("Full PDF text", text, height=260)

            if mode == "Ask a question":
                question = st.text_input(
                    "Ask about the PDF",
                    placeholder="Try: enrollment number, name, or another detail",
                )
                if question:
                    q = question.lower()
                    st.subheader("Answer")
                    if "enrollment" in q:
                        match = re.search(
                            r"Enrollment\s*(?:No|Number|ID)?[:\s]*([A-Za-z0-9]+)",
                            text,
                            re.IGNORECASE,
                        )
                        st.write(
                            match.group(1) if match else "Enrollment number not found."
                        )
                    elif "name" in q:
                        match = re.search(
                            r"Name\s*(?:[:\-]|is)?\s*([A-Za-z\s]+)",
                            text,
                            re.IGNORECASE,
                        )
                        st.write(match.group(1).strip() if match else "Name not found.")
                    else:
                        nearest = nearest_chunks(question, chunks, embeddings, top_k=3)
                        if nearest:
                            st.write(
                                "No exact answer found. Here are the nearest relevant text segments:"
                            )
                            for i, (score, snippet) in enumerate(nearest, start=1):
                                st.markdown(
                                    f"**Segment {i}** — similarity: {score:.3f}\n\n{snippet}"
                                )
                        else:
                            st.write(
                                "No relevant content found. Try a different query or use the keyword search."
                            )
            else:
                summary_length = st.slider("Summary length", 2, 10, 5)
                st.subheader("Document summary")
                st.write(summarize_text(text, summary_length))
