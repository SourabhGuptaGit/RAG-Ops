from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from typing import List


load_dotenv()

llm_client = OpenAI(timeout=30)
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072


splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=150)

def load_and_chunk_pdf(file_path: str) -> List[str]:
    docs = PDFReader().load_data(file=file_path)
    texts = [doc.text for doc in docs if getattr(doc, "text", None)]
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    response = llm_client.embeddings.create(
        model=EMBED_MODEL,
        dimensions=EMBED_DIM,
        input=texts
    )
    
    return [item.embedding for item in response.data]
