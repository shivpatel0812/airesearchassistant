from fastapi import FastAPI
from pydantic import BaseModel

from data_ingest import load_texts, chunk_document
from indexing import VectorIndex
from inference.rag import load_model, generate_answer

app = FastAPI()

class Query(BaseModel):
    question: str


@app.on_event('startup')
def startup_event():
    global index, tokenizer, model
    docs = []
    for doc in load_texts('corpus'):
        docs.extend(list(chunk_document(doc)))
    index = VectorIndex()
    if docs:
        index.build(docs)
    tokenizer, model = load_model()


@app.post('/ask')
def ask(query: Query):
    answer, passages = generate_answer(query.question, index, tokenizer, model)
    return {
        'answer': answer,
        'sources': [{'source': p['doc'].source, 'score': p['score']} for p in passages]
    }
