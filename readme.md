# Bioinformatics RAG System

A Retrieval-Augmented Generation (RAG) system that answers questions from PubMed research papers on single-cell RNA sequencing (scRNA-seq).

The system uses BioBERT for embeddings, FAISS for retrieval, and a local LLM (Gemma via Ollama) to generate answers strictly from the provided papers.

---

## Overview

- Takes a user query
- Retrieves relevant text from ~15 research papers
- Generates an answer based only on those papers

No external knowledge is used.

---

## Pipeline

User Query  
→ BioBERT Embedding  
→ FAISS Retrieval  
→ Relevant Text Chunks  
→ LLM (Gemma)  
→ Answer + References  

---

## Features

- Domain-specific QA system (bioinformatics)
- BioBERT-based embeddings
- FAISS vector search
- Local LLM (Gemma via Ollama)
- Streamlit UI
- Answers with references
- Retrieved context visibility

---

## Project Structure

BioRAG/
│
├── data/
│   ├── embedded_data.json
│   ├── faiss_index.bin
│
├── papers/
│   ├── *.pdf
│
├── app.py
├── requirements.txt
└── README.md

---

## Installation

git clone <your-repo-link>  
cd BioRAG  
pip install -r requirements.txt  

---

## Run

streamlit run app.py  

---

## Example Query

What is single cell RNA sequencing?

---

## Optional GPU Setup

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  

---

## Limitations

- Answers depend only on the provided papers  
- Retrieval quality depends on embeddings  
- Small LLM may produce short answers  

---

## Tech Stack

Python  
PyTorch  
Transformers (BioBERT)  
FAISS  
Ollama (Gemma)  
Streamlit  
PyMuPDF  

---

## Author

Poonam Suwasia