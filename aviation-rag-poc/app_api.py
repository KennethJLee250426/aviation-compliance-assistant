import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel

app = FastAPI(title="Aviation Regulatory RAG POC")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_DIR = "./regulatory_chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists(DB_DIR):
  vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
else:
  vectorstore = None

llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.0)


class QueryRequest(BaseModel):
  question: str
  authority: str = "ALL"


@app.get("/", response_class=HTMLResponse)
async def read_index():
  if os.path.exists("templates/index.html"):
    with open("templates/index.html", "r", encoding="utf-8") as f:
      return f.read()
  return "<h3>Error: templates/index.html not found!</h3>"


@app.post("/api/query")
async def query_rag(req: QueryRequest):
  if not vectorstore:
    raise HTTPException(
        status_code=400,
        detail="Vector database not found. Please run ingest.py first.",
    )

  search_kwargs = {"k": 4}
  if req.authority != "ALL":
    search_kwargs["filter"] = {"authority": req.authority}

  retriever = vectorstore.as_retriever(
      search_type="similarity", search_kwargs=search_kwargs
  )

  template = """You are an expert Quality, Engineering, Health, and Safety (QEHS) regulatory compliance specialist.
Answer the question accurately using only the provided regulatory context. Cite the specific authority and document source referenced.

Context:
{context}

Question: {question}
"""
  prompt = ChatPromptTemplate.from_template(template)

  def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('authority', 'UNKNOWN')} - {os.path.basename(doc.metadata.get('source', 'Unknown'))}]\n{doc.page_content}"
        for doc in docs
    )

  try:
    retrieved_docs = retriever.invoke(req.question)
    context_text = format_docs(retrieved_docs)

    formatted_prompt = prompt.invoke(
        {"context": context_text, "question": req.question}
    )
    response = llm.invoke(formatted_prompt)

    return {"answer": response.content, "sources": context_text}
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
  import uvicorn

  uvicorn.run("app_api:app", host="0.0.0.0", port=8000, reload=True)