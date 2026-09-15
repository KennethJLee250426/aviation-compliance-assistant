import streamlit as st
import torch
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 1. Page Configuration
st.set_page_config(
    page_title="EASA Part-145 Audit Assistant",
    page_icon="✈️",
    layout="wide"
)

# 2. Cached RAG Pipeline Loader
@st.cache_resource
def load_rag_pipeline():
    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Vector Database
    vector_db = Chroma(
        persist_directory="./easa_chroma_db", 
        embedding_function=embeddings
    )
    
    # LLM & Tokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    )
    llm = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    # Cross-Encoder Re-ranker
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    
    return vector_db, llm, reranker, tokenizer

# Initialize Models
with st.spinner("Initializing EASA Compliance Engine & Neural Models..."):
    vector_db, llm, reranker, tokenizer = load_rag_pipeline()

# 3. User Interface Header
st.title("✈️ EASA Part-145 Audit & Compliance Assistant")
st.markdown("Automated regulatory retrieval and finding synthesis for MRO quality assurance.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Audit Settings")
    retrieval_k = st.slider("Vector Retrieval Depth (k)", min_value=3, max_value=10, value=5)
    rerank_top_n = st.slider("Top Reranked Passages", min_value=1, max_value=5, value=3)
    st.divider()
    st.info("Status: Vector Store Online & Model Loaded")

# 4. User Input Area
query = st.text_area(
    "Compliance Query / Audit Scenario", 
    placeholder="e.g., What are the qualification requirements for Component Certifying Staff (CC/S) under Part-145?",
    height=100
)

# 5. Query Execution Logic
if st.button("Execute Audit Inspection", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a valid regulatory query before executing.")
    else:
        with st.spinner("Retrieving regulatory specification chunks and reranking..."):
            # Step A: Hybrid Keyword Query Expansion & Similarity Search
            expanded_query = f"{query} EASA Part-145 145.A.30 145.A.35 145.A.45 145.A.50 CC/S MOE CRS Form 1"
            raw_docs = vector_db.similarity_search(expanded_query, k=retrieval_k)
            
            # Step B: Cross-Encoder Reranking
            if raw_docs:
                pairs = [[query, doc.page_content] for doc in raw_docs]
                scores = reranker.predict(pairs)
                scored_docs = sorted(zip(scores, raw_docs), key=lambda x: x[0], reverse=True)
                top_docs = [doc for score, doc in scored_docs[:rerank_top_n]]
            else:
                top_docs = []
            
            # Step C: Prompt Construction
            if top_docs:
                context_blocks = [f"Passage {i}:\n{doc.page_content}" for i, doc in enumerate(top_docs, start=1)]
                context_str = "\n\n".join(context_blocks)
            else:
                context_str = "No specific regulatory context retrieved."

            prompt = (
                f"<|im_start|>system\n"
                f"You are an expert EASA Quality Assurance Auditor.\n"
                f"Answer the audit query clearly based strictly on the regulatory context provided below.\n"
                f"If the context does not contain enough information, state what is missing.<|im_end|>\n"
                f"<|im_start|>user\nREGULATORY CONTEXT:\n{context_str}\n\nAUDIT QUERY:\n{query}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            # Step D: LLM Generation
            raw_response = llm(
                prompt, 
                max_new_tokens=400, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )[0]["generated_text"]
            
            clean_response = raw_response[len(prompt):].strip()

        # 6. Render Dual Column Results
        col1, col2 = st.columns([1, 1], gap="medium")
        
        with col1:
            st.subheader("📋 Audit Finding & Compliance Guidance")
            st.write(clean_response)
            
        with col2:
            st.subheader("🔍 Retained Source Citations")
            if top_docs:
                for i, doc in enumerate(top_docs, start=1):
                    meta = getattr(doc, "metadata", {}) or {}
                    
                    # Search across all potential metadata keys for filename
                    source_val = (
                        meta.get("source") or 
                        meta.get("filename") or 
                        meta.get("file_path") or 
                        meta.get("title") or 
                        f"EASA Spec Chunk #{i}"
                    )
                    
                    # Format clean filename
                    clean_source = str(source_val).replace("\\", "/").split("/")[-1]
                    
                    # Render citation box
                    with st.expander(f"Source {i}: {clean_source}", expanded=True):
                        st.write(doc.page_content)
            else:
                st.warning("No regulatory passages matched this query.")