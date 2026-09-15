import streamlit as st
import torch
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

st.set_page_config(page_title="EASA Part-145 Auditor", layout="wide")
st.title("🔍 EASA Quality & Compliance Engine")

@st.cache_resource
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./easa_chroma_db", embedding_function=embeddings)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    )
    llm = pipeline("text-generation", model=model, tokenizer=tokenizer)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    
    return vector_db, llm, reranker, tokenizer

vector_db, llm, reranker, tokenizer = load_rag_pipeline()

query = st.text_area(
    "Compliance Query / Audit Scenario", 
    placeholder="e.g., Can a CoC be accepted in lieu of an EASA Form 1 for component installation?",
    height=120
)

if st.button("Execute Audit Inspection", type="primary"):
    if not query.strip():
        st.warning("Please enter a valid regulatory query.")
    else:
        with st.spinner("Searching regulatory database and reranking..."):
            expanded_query = f"{query} EASA Part-145 145.A.45 145.A.50 CRS Form 1 ARC MOE"
            raw_docs = vector_db.similarity_search(expanded_query, k=5)
            
            pairs = [[query, doc.page_content] for doc in raw_docs]
            scores = reranker.predict(pairs)
            scored_docs = sorted(zip(scores, raw_docs), key=lambda x: x[0], reverse=True)
            top_docs = [doc for score, doc in scored_docs[:3]]
            
            context_blocks = [f"Passage {i} [{doc.metadata.get('source', 'EASA Spec')}]:\n{doc.page_content}" for i, doc in enumerate(top_docs, start=1)]
            context_str = "\n\n".join(context_blocks)
            
            prompt = (
                f"<|im_start|>system\n"
                f"You are an expert EASA Quality Engineering & Part-145 Airworthiness Auditor.\n"
                f"Acronym Definitions:\n"
                f"- ARC = Authorized Release Certificate (EASA Form 1)\n"
                f"- CoC = Certificate of Conformity\n"
                f"- D1 = Class D1 Specialised Services (NDT)\n"
                f"- CC/S = Component Certifying Staff\n"
                f"- SPM = Standard Practice Manual\n"
                f"- MOE = Maintenance Organisation Exposition\n\n"
                f"Answer the query accurately based strictly on the context provided.<|im_end|>\n"
                f"<|im_start|>user\nREGULATORY CONTEXT:\n{context_str}\n\nAUDIT QUERY:\n{query}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            generation = llm(
                prompt, 
                max_new_tokens=400, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )[0]["generated_text"]
            
            response = generation[len(prompt):].strip()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📋 Audit Finding & Guidance")
            st.write(response)
            
        with col2:
            st.subheader("🔍 Source Citations")
            for i, doc in enumerate(top_docs, start=1):
                src = doc.metadata.get("source", "EASA Specification")
                st.info(f"**Source {i} [{src}]:**\n\n{doc.page_content}")