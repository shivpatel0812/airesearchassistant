from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable

@dataclass
class Document:
    text: str
    source: str


def load_texts(directory: str) -> List[Document]:
    """Load .txt files from a directory."""
    docs = []
    for path in Path(directory).rglob('*.txt'):
        text = path.read_text(encoding='utf-8')
        docs.append(Document(text=text, source=str(path)))
    return docs


def chunk_document(doc: Document, chunk_size: int = 500, overlap: int = 50) -> Iterable[Document]:
    """Yield overlapping chunks from a document."""
    words = doc.text.split()
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_text = ' '.join(words[start:end])
        yield Document(text=chunk_text, source=doc.source)
        start = end - overlap
