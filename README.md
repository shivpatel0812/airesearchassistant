# Domain-Aware Research Assistant (Skeleton)

This repository contains a minimal prototype of a research assistant using Retrieval-Augmented Generation (RAG) with the Mistral 7B model. The pipeline covers:

1. **Data ingestion** from a directory of text files.
2. **Vector indexing** of text chunks with FAISS and Sentence Transformers.
3. **RAG inference** that retrieves relevant passages and generates an answer.
4. **Optional LoRA fine-tuning** utilities using `peft`.
5. **FastAPI web app** exposing a `/ask` endpoint.

## Usage

1. Place your `.txt` documents in the `corpus/` directory.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the API server:

```bash
uvicorn webapp.app:app --reload
```

Send a POST request to `/ask` with a JSON body `{"question": "..."}`.

Model weights are loaded from Hugging Face and may require a GPU for efficient inference.
