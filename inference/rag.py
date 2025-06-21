from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_ingest import Document
from indexing import VectorIndex


def load_model(model_name: str = 'mistralai/Mistral-7B-v0.1'):
    """Load Mistral model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


def format_prompt(question: str, passages: List[Document]) -> str:
    snippet_text = "\n\n".join(f"[Source: {p.source}]\n{p.text}" for p in passages)
    prompt = (
        "You are an expert research assistant. Use the passages below to answer the question.\n"
        f"{snippet_text}\n\nQuestion: {question}\nAnswer:"
    )
    return prompt


def generate_answer(question: str, index: VectorIndex, tokenizer, model, k: int = 3):
    results = index.query(question, k)
    passages = [r['doc'] for r in results]
    prompt = format_prompt(question, passages)
    inputs = tokenizer(prompt, return_tensors='pt')
    output = model.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer, results
