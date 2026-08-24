"""
=============================================================================
Case 1: Intelligent Customer Service and Enterprise Private Knowledge Base System (RAG Engine Core Module)
-----------------------------------------------------------------------------
[Teaching Scope]:
This module implements the complete RAG (Retrieval-Augmented Generation) pipeline covered on slides 10–37.
The pipeline has been refactored using the modern LCEL (LangChain Expression Language) syntax.

[Key Features]:
1. Embedding layer: Supports the free cloud-based SiliconFlow BAAI/bge-m3 API, eliminating the need to download the 1.2 GB model and providing fast responses;
2. Fallback support: Supports offline Hugging Face BAAI/bge-m3 / bge-small loading with a Mock fallback;
3. Chunking and retrieval: Uses the standard 800/100-character chunking strategy, Top-K=4 retrieval, and persistent local FAISS vector storage.
=============================================================================
"""

import os
import logging
from typing import List, Tuple, Any, Optional

# Load API key and model configuration from .env
try:
    from dotenv import load_dotenv
    _env_candidates = []
    if os.getenv("DOTENV_PATH"):
        _env_candidates.append(os.getenv("DOTENV_PATH"))
    _here = os.path.dirname(os.path.abspath(__file__))
    _env_candidates.append(os.path.join(_here, ".env"))
    _env_candidates.append(os.path.join(os.path.dirname(_here), "runtime", ".env"))
    _env_candidates.append(os.path.join(os.path.dirname(_here), ".env"))
    for _p in _env_candidates:
        if os.path.exists(_p):
            load_dotenv(_p)
            break
except ImportError:
    pass

# --- 1. PDF parsing and text splitting ---
try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        PdfReader = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        # Pure-Python recursive text splitter used as a teaching fallback
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=800, chunk_overlap=100, separators=None):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
            def split_text(self, text: str):
                chunks = []
                start = 0
                while start < len(text):
                    end = start + self.chunk_size
                    chunks.append(text[start:end])
                    start += (self.chunk_size - self.chunk_overlap)
                    if start >= len(text):
                        break
                return chunks

# --- 2. Embedding and retrieval libraries ---
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    FAISS = None

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import OpenAIEmbeddings
    except ImportError:
        OpenAIEmbeddings = None

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    HuggingFaceEmbeddings = None

# --- 3. Modern LCEL chain components ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Default SiliconFlow API configuration
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")


# -----------------------------------------------------------------------------
# Step 1: Load and parse PDF documents (corresponds to Slide 13)
# -----------------------------------------------------------------------------
def pdf_read(pdf_docs: List[Any]) -> str:
    """Extract plain text from one or more PDF file objects"""
    if PdfReader is None:
        raise ImportError("Please install a PDF parsing library first: pip install pypdf or pip install PyPDF2")
        
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page_idx, page in enumerate(pdf_reader.pages):
                content = page.extract_text()
                if content:
                    text += content + "\n"
        except Exception as e:
            logging.warning(f"⚠️ Failed to parse PDF file: {e}")
            
    if not text.strip():
        raise ValueError("No valid text could be extracted from the PDF. Please make sure the PDF is not a scanned image-only document.")
        
    return text


# -----------------------------------------------------------------------------
# Step 2: Split long text into intelligent chunks (corresponds to Slide 14)
# -----------------------------------------------------------------------------
def get_chunks(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> List[str]:
    """Use a recursive character splitter to divide long text into chunks suitable for embedding (course standard: 800/100)"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks


# -----------------------------------------------------------------------------
# Step 3: Load the embedding model (SiliconFlow BGE-M3 online API preferred; corresponds to Slide 15)
# -----------------------------------------------------------------------------
def get_embeddings(model_name: str = "BAAI/bge-m3"):
    """
    Get the embedding model:
    1. Prefer the free cloud-based SiliconFlow BAAI/bge-m3 (millisecond-level response without downloading the 1.2 GB model);
    2. Fall back to HuggingFaceEmbeddings for offline use;
    """
    # Prefer the online API
    if OpenAIEmbeddings is not None and SILICONFLOW_API_KEY:
        try:
            embeddings = OpenAIEmbeddings(
                model=model_name,
                openai_api_key=SILICONFLOW_API_KEY,
                openai_api_base=SILICONFLOW_BASE_URL,
                check_embedding_ctx_length=False
            )
            return embeddings
        except Exception as e:
            logging.warning(f"⚠️ SiliconFlow online embedding initialization failed: {e}，trying the local fallback。")

    # Local offline model fallback
    if HuggingFaceEmbeddings is not None:
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    raise ImportError("Please install langchain-openai or sentence-transformers: pip install langchain-openai faiss-cpu")


# -----------------------------------------------------------------------------
# Step 4: Build and persist the vector database (corresponds to Slide 15)
# -----------------------------------------------------------------------------
def create_vector_store(
    text_chunks: List[str],
    embeddings=None,
    save_path: str = "faiss_db"
):
    """Build a FAISS index in memory and persist it to local disk"""
    if FAISS is None:
        raise ImportError("Please install FAISS first: pip install faiss-cpu")
        
    if embeddings is None:
        embeddings = get_embeddings()
        
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local(save_path)
    return vector_store


def load_vector_store(load_path: str = "faiss_db", embeddings=None):
    """Load the pre-built FAISS vector store from local disk"""
    if FAISS is None:
        raise ImportError("Please install FAISS first: pip install faiss-cpu")
        
    if embeddings is None:
        embeddings = get_embeddings()
        
    return FAISS.load_local(
        load_path,
        embeddings,
        allow_dangerous_deserialization=True
    )


# -----------------------------------------------------------------------------
# Step 5: Assemble the modern LCEL retrieval and QA chain (refactored from the legacy RetrievalQA; corresponds to Slides 17–19)
# -----------------------------------------------------------------------------
LANGUAGE_INSTRUCTIONS = {
    "English":
        "Always answer in English.",

    "Chinese":
        "Always answer in Chinese.",

    "Bilingual":
        "Answer in both Chinese and English. "
        "First provide Chinese, then provide English translation."
}
RAG_PROMPT_TEMPLATE = """You are a professional enterprise knowledge assistant. Answer the user's [Question] based only on the [Known Information] below.
If the answer is not contained in the known information, reply directly with "The document content does not provide an answer to this question." Do not fabricate an answer.

[Known Information]:
{context}

[Question]:
{question}

Answer Language:
{language_instruction}

Rules:
1. Only use information from the retrieved context when answering.
2. If the context does not contain enough information, clearly state that.
3. Keep the answer professional and concise.
4. Follow the requested output language strictly.

Answer:
"""


def format_docs(docs) -> str:
    """Helper function: format retrieved document chunks"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        formatted.append(f"[Chunk {i}]:\n{content}")
    return "\n\n".join(formatted)


def build_rag_chain(retriever, llm):
    """Build a clean, efficient LCEL chain that supports streaming inference"""
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
