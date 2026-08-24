"""
=============================================================================
Case 1: Enterprise Private Knowledge Base Streamlit Web Application
-----------------------------------------------------------------------------
Teaching Positioning：
This module implements the Streamlit frontend interface presented on Slides 20–21 of the course materials.
The left sidebar handles document uploads and vector-store index maintenance;
the main area handles interactive Q&A, streamed Qwen/DeepSeek responses, and expandable source citations.

Model Configuration：
- Embedding: SiliconFlow cloud free BAAI/bge-m3 (1024 dimensions)
- LLM: SiliconFlow cloud free Qwen/Qwen3-8B (with automatic fallback to ChatOllama / Mock)
=============================================================================
"""

import os
import time
import logging
import streamlit as st

# Load the API key and model configuration from .env.
# Prefer DOTENV_PATH passed by the .bat launcher; otherwise search the local and runtime directories.
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

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import the core RAG modules
from rag_engine_en import (
    pdf_read,
    get_chunks,
    get_embeddings,
    create_vector_store,
    load_vector_store,
    build_rag_chain,
    format_docs,
    RAG_PROMPT_TEMPLATE,
    LANGUAGE_INSTRUCTIONS,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL
)

# Basic page configuration (course Slide 20 style)
st.set_page_config(
    page_title="Enterprise Private Knowledge Base - Local RAG",
    page_icon="🏢",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 20px;
        color: #1E293B;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 1. State Management and Initialization
# -----------------------------------------------------------------------------
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(_APP_DIR, "faiss_db")

def is_valid_db(path: str = DB_DIR) -> bool:
    """Strictly validate the integrity of the local FAISS index files."""
    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, "index.faiss"))
        and os.path.exists(os.path.join(path, "index.pkl"))
    )

if "db_ready" not in st.session_state:
    st.session_state.db_ready = is_valid_db()

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------------------------------------------------------
# 2. Embedding Model and LLM Singleton Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_cached_embeddings():
    """Load and cache the BGE-M3 embedding model (prefer SiliconFlow cloud)."""
    return get_embeddings("BAAI/bge-m3")


def get_llm():
    """
    Get a large language model instance:
    1. Prefer the free Qwen/Qwen3-8B model on SiliconFlow cloud;
    2. 2. Fall back to local ChatOllama (deepseek-r1:1.5b);
    3. 3. Use FakeListLLM as the final teaching/demo fallback.
    """
    # 1. Prefer the free cloud API
    if SILICONFLOW_API_KEY:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=os.getenv("ONLINE_LLM_MODEL", "Qwen/Qwen3-8B"),
                openai_api_key=SILICONFLOW_API_KEY,
                openai_api_base=SILICONFLOW_BASE_URL,
                temperature=0.1,
                streaming=True,
                extra_body={"enable_thinking": False}
            )
        except Exception as e:
            logging.warning(f"⚠️ Cloud LLM connection failed: {e}. Trying the local model.")

    # 2. Fall back to local Ollama
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model="deepseek-r1:1.5b", temperature=0.1)
    except Exception:
        try:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(model="deepseek-r1:1.5b", temperature=0.1)
        except Exception:
            # 3. Final Mock fallback
            from langchain_core.language_models.fake import FakeListLLM
            return FakeListLLM(responses=[
                "Demo system response: According to the enterprise knowledge base, the key advantage of local LLM deployment is keeping data within the organization. LangChain and BGE-M3 vectorization can also support highly available, enterprise-grade question answering and retrieval."
            ])


# -----------------------------------------------------------------------------
# 3. Sidebar: Document Management and Knowledge Base Construction (corresponding to the left side of Slide 20)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📁 Document Management")
    st.markdown("---")

    answer_language = st.selectbox(
        "🌐 Answer Language",
        [
            "English",
            "Chinese",
            "Bilingual"
        ],
        index=0
    )
    # Dynamic status container (using st.empty to refresh the status immediately after a successful build)
    status_placeholder = st.empty()
    if st.session_state.db_ready:
        status_placeholder.success("✅ Knowledge base status: Ready")
    else:
        status_placeholder.warning("⚠️ Knowledge base status: Not built (please upload documents)")
        
    st.markdown("---")
    
    # PDF upload component
    uploaded_files = st.file_uploader(
        "Upload internal PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Supports multiple uploads of corporate policies, technical specifications, or research reports"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("🚀 Submit and Process", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear Knowledge Base", use_container_width=True)
        
    # Process uploaded files and build vector embeddings
    if process_btn:
        if not uploaded_files:
            st.error("Please select at least one PDF file first! (You may use the sample files in the sample_docs/ directory.)")
        else:
            with st.spinner("Parsing PDFs, splitting long text (800/100), and building the vector index with BGE-M3..."):
                try:
                    raw_text = pdf_read(uploaded_files)
                    chunks = get_chunks(raw_text, chunk_size=800, chunk_overlap=100)
                    st.info(f"📄 Document extraction complete. The content has been intelligently split into {len(chunks)} chunks.")
                    
                    embeddings = load_cached_embeddings()
                    create_vector_store(chunks, embeddings=embeddings, save_path=DB_DIR)
                    
                    st.session_state.db_ready = True
                    # Immediately refresh the status badge and notification
                    status_placeholder.success("✅ Knowledge base status: Ready")
                    st.toast("🎉 Knowledge base built successfully. Ready to use.", icon="✅")
                    st.success("🎉 Vector database built successfully. You can start asking questions on the right.")
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
                    
    # Clear Knowledge Base
    if clear_btn:
        if os.path.exists(DB_DIR):
            import shutil
            shutil.rmtree(DB_DIR, ignore_errors=True)
        st.session_state.db_ready = False
        status_placeholder.warning("⚠️ Knowledge base status: Not built")
        st.session_state.messages = []
        st.toast("🗑️ Knowledge base cleared", icon="ℹ️")
        st.rerun()


# -----------------------------------------------------------------------------
# 4. Main Interface: Intelligent Customer Service Q&A (corresponding to Slide 21)
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🏢 Enterprise Private Knowledge Base (Local RAG)</div>', unsafe_allow_html=True)
st.caption("A modern document Q&A system built with BAAI/bge-m3 + Qwen/Qwen3-8B + LangChain LCEL + FAISS")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View referenced document excerpts"):
                for i, doc in enumerate(msg["sources"], 1):
                    content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                    st.markdown(f"**Source excerpt {i}:**\n```text\n{content}\n```")

# Receive user input
user_query = st.chat_input("Enter a business or document question...")

if user_query:
    if not st.session_state.db_ready:
        st.error("The knowledge base has not been built yet. Please upload PDF documents on the left and click 'Submit and Process'.")
    else:
        # Display the user's question
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)
            
        # Run RAG retrieval and generate the streamed response
        with st.chat_message("assistant"):
            try:
                with st.spinner("AI is searching the knowledge base..."):
                    embeddings = load_cached_embeddings()
                    vector_store = load_vector_store(DB_DIR, embeddings=embeddings)
                    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                    source_docs = retriever.invoke(user_query)
                
                llm = get_llm()
                context_text = format_docs(source_docs)
                prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
                
                # Assemble the efficient streaming LCEL chain
                stream_chain = prompt_template | llm | StrOutputParser()
                
                # Use Streamlit's native typewriter-style streaming output
                response = st.write_stream(stream_chain.stream({
                    "context": context_text,
                    "question": user_query,
                    "language_instruction":
                        LANGUAGE_INSTRUCTIONS[answer_language]
                }))
                
                # Expandable source traceability card
                if source_docs:
                    with st.expander("🔍 View referenced document excerpts"):
                        for i, doc in enumerate(source_docs, 1):
                            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                            st.markdown(f"**Source excerpt {i}:**\n```text\n{content}\n```")
                            
                # Record the response in conversation history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": source_docs
                })
            except Exception as e:
                err_msg = f"Retrieval and generation failed: {str(e)}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
