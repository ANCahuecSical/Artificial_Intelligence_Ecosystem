
# Logging and warnings configuration 🔧
import logging
from transformers import logging as hf_logging
import warnings

# Set log level for langchain.text_splitter and transformers to ERROR
logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
# Also set transformers' internal verbosity to ERROR
hf_logging.set_verbosity_error()

# Suppress Python warnings
warnings.filterwarnings("ignore")
import os
# Load environment variables from .env and set OpenAI API key 🔐
from dotenv import load_dotenv
from pathlib import Path

# Load .env from the same directory as this script
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simple .env test utility for manual verification 🔍
def test_openai_key() -> bool:
    """Return True if OPENAI_API_KEY is set and print a masked confirmation."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        masked = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "(masked)"
        print(f"OPENAI_API_KEY loaded: {len(api_key)} chars, masked: {masked}")
        return True
    else:
        print("ERROR: OPENAI_API_KEY not set. Please provide a .env with OPENAI_API_KEY.")
        return False






# Chunking and embedding configuration
chunk_size = 500
chunk_overlap = 50
model_name = "sentence-transformers/all-distilroberta-v1"
top_k = 20

# Re-ranking configuration
cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
top_m = 8

# Read selected document into `text` (UTF-8)
selected_path = "Selected_Document.txt"
try:
    with open(selected_path, "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    # Provide a clear error if the test file is missing
    raise FileNotFoundError(f"Selected document not found: {selected_path}")

# Split text into chunks using RecursiveCharacterTextSplitter ✅
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Use separators ['', '\n', ' ', ''] and the configured chunk_size / chunk_overlap
splitter = RecursiveCharacterTextSplitter(
    separators=['', '\n', ' ', ''],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)
chunks = splitter.split_text(text)

# Encode chunks with SentenceTransformer and build a FAISS index 🔁
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Load the sentence-transformer model
sbert = SentenceTransformer(model_name)
# Encode with a visible progress bar
embeddings = sbert.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

# Ensure embeddings is two-dimensional and float32
if embeddings.ndim == 1:
    embeddings = embeddings.reshape(1, -1)
embeddings = embeddings.astype(np.float32)

# Initialize FAISS index with the correct dimension and add embeddings
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Provide canonical names expected by retrieval utilities
embedder = sbert
faiss_index = index

# Retrieval helper: encode a question and return top-k chunks
def retrieve_chunks(question: str, k: int = top_k):
    """Encode `question` and return the top-k matching chunks.

    Uses the variables `embedder`, `faiss_index`, and `chunks` from the module scope.
    Returns a list of text chunks in descending order of relevance (closest first).
    """
    # Encode the question (follow the specified call signature)
    q_vec = embedder.encode([question], show_progress_bar=False)

    # Convert to a NumPy float32 array with shape (1, dim)
    q_arr = np.array(q_vec, dtype=np.float32)
    if q_arr.ndim == 1:
        q_arr = q_arr.reshape(1, -1)

    # Search the FAISS index for the top-k neighbors
    _, I = faiss_index.search(q_arr, k)

    # I is an array of indices; return the corresponding chunks
    indices = I[0].tolist()
    return [chunks[i] for i in indices]

# Cross-encoder re-ranking utilities 🔁
from sentence_transformers import CrossEncoder
import re

# Initialize the cross-encoder model
reranker = CrossEncoder(cross_encoder_name)


def dedupe_preserve_order(items):
    """Return a list with duplicates removed while preserving first occurrence.

    Whitespace is normalized to avoid near-duplicate slices.
    """
    seen = set()
    out = []
    for it in items:
        norm = re.sub(r"\s+", " ", it).strip()
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def rerank_chunks(question: str, candidate_chunks: list[str], m: int = top_m) -> list[str]:
    """Rerank `candidate_chunks` for `question` using a CrossEncoder and return top-m chunks.

    This function ONLY uses the cross-encoder for scoring and does not re-encode with the bi-encoder.
    """
    # Create (question, chunk) pairs for the cross-encoder
    pairs = [(question, chunk) for chunk in candidate_chunks]

    # Score pairs (higher = more relevant)
    scores = reranker.predict(pairs)

    # Sort chunks by score (descending) and take top-m
    scored = list(zip(scores, candidate_chunks))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [c for _, c in scored[:m]]

    # Light dedupe / normalize whitespace
    return dedupe_preserve_order(top_chunks)

from typing import List, Tuple


def build_prompts(context: str, question: str) -> Tuple[str, str]:
    """Build system and user prompts for the chat completion.

    Returns a (system_prompt, user_prompt) tuple. The user prompt contains the
    provided context and the question, and asks the assistant to answer.
    """
    system_prompt = (
        "You are a knowledgeable assistant that answers questions based on the provided context. "
        "If the answer is not in the context, say you don't know."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return system_prompt, user_prompt


def answer_question(question: str) -> str:
    """Answer `question` using retrieval, reranking, and an LLM chat completion."""
    # Retrieve bi-encoder candidates
    candidates: List[str] = retrieve_chunks(question)

    # Re-rank with the cross-encoder
    relevant_chunks: List[str] = rerank_chunks(question, candidates, m=top_m)

    # Build context
    context = "\n\n".join(relevant_chunks)

    # Build prompts
    system_prompt, user_prompt = build_prompts(context, question)

    # Call OpenAI (NEW SDK)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=500,
    )

    return resp.choices[0].message.content.strip()





def main() -> None:
    if not test_openai_key():
        raise SystemExit(1)

    print("Enter 'exit' or 'quit' to end.")
    while True:
        try:
            question = input("Your question: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.strip().lower() in ("exit", "quit"):
            break

        try:
            answer = answer_question(question)
            print("Answer:", answer)
        except Exception as e:
            print("Error while answering question:", e)


if __name__ == "__main__":
    main()
