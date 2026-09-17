# Aviation Compliance Assistant

A local, privacy-preserving RAG (Retrieval-Augmented Generation) application for querying multi-authority aviation regulatory documents — EASA, CAAS, and CAAC — using a locally hosted LLM.

## Overview

This tool lets you ask natural-language compliance questions and get answers grounded in actual regulatory source documents, with citations back to the specific authority and file. All inference runs locally via [Ollama](https://ollama.com) — no data leaves your machine, making it suitable for querying sensitive or internal compliance documentation alongside public regulatory texts.

## Features

- **Multi-authority filtering** — scope queries to EASA, CAAS, CAAC, or search across all
- **Source-grounded answers** — responses are generated only from retrieved regulatory context, with citations
- **Fully local** — no external API calls; runs on local Ollama models (tested with Qwen2.5)
- **Simple web UI** — lightweight chat interface, no framework dependencies

## Tech Stack

- **Backend:** FastAPI
- **LLM inference:** Ollama (Qwen2.5)
- **Vector store:** Chroma
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
- **Orchestration:** LangChain
- **Frontend:** Vanilla HTML/JS

## Project Structure
aviation-rag-poc/
├── app_api.py # FastAPI backend + query endpoint
├── ingest.py # Document ingestion & vector DB builder
├── templates/
│ └── index.html # Chat UI
├── regulations/
│ ├── easa/ # EASA source documents
│ ├── caas/ # CAAS source documents
│ └── caac/ # CAAC source documents
├── regulatory_chroma_db/ # Generated vector store (gitignored)
└── requirements.txt

## Data Sources

Regulatory documents in the `regulations/` directory were sourced from the official public websites of the respective authorities:

- **EASA** — easa.europa.eu
- **CAAS** — caas.gov.sg
- **CAAC** — caac.gov.cn

These documents remain subject to the original terms of use published by their respective authorities. This repository's MIT license applies only to the application code (`app_api.py`, `ingest.py`, `templates/`), not to the regulatory documents themselves.


## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed locally
- A pulled Ollama model (e.g. `ollama pull qwen2.5:7b`)

### Installation

```bash
git clone https://github.com/KennethJLee250426/aviation-compliance-assistant
cd aviation-compliance-assistant/aviation-rag-poc
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Build the vector database

```bash
python ingest.py
```

### Run the app

```bash
ollama serve  # in a separate terminal, if not already running
uvicorn app_api:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

## Notes

- Regulatory source documents included in `regulations/` are publicly published materials from EASA, CAAS, and CAAC.
- Model choice is configurable in `app_api.py` — smaller models (e.g. `qwen2.5:1.5b`) run faster on CPU-only environments; larger models (`qwen2.5:7b`) benefit significantly from GPU acceleration.

## Status

Proof-of-concept, under active development.
