import os
from dotenv import load_dotenv

config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.env")
load_dotenv(config_path)

import chromadb
from chromadb.utils import embedding_functions
import asyncio

chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Using HuggingFace API key through environment variable (HF_TOKEN) automatically handled by transformers
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(name="its_tv_scripts", embedding_function=emb_fn)

def ingest_scripts(directory: str):
    """
    Memproses folder PDF/teks skrip ITS TV lama menjadi embeddings di ChromaDB.
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} tidak ditemukan.")
        return
    
    docs = []
    ids = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Chunking sederhana dengan pemisah paragraf
                chunks = [chunk.strip() for chunk in content.split("\n\n") if len(chunk.strip()) > 50]
                for i, chunk in enumerate(chunks):
                    docs.append(chunk)
                    ids.append(f"{filename}_chunk_{i}")
                    
    if docs:
        collection.upsert(documents=docs, ids=ids)
        print(f"Berhasil menambahkan {len(docs)} segmen dari {directory} ke vector DB.")

def _query_rag_sync(scene_text: str) -> str:
    try:
        results = collection.query(
            query_texts=[scene_text],
            n_results=3 # Mengambil 3 dokumen agar konteks visual lebih kaya
        )
        if results and 'documents' in results and results['documents'] and len(results['documents'][0]) > 0:
            return " ".join(results['documents'][0])
    except Exception as e:
        print(f"RAG Error: {e}")
    return ""

async def get_visual_context(scene_text: str) -> str:
    """
    Retrieves relevant text segments from ChromaDB based on a query.
    Used to ground the scene generation within the context of the uploaded script.
    """
    print(f"Retrieving visual context for: {scene_text}")
    return await asyncio.to_thread(_query_rag_sync, scene_text)
