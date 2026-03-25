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

def _query_rag_sync(scene_text: str) -> list:
    try:
        results = collection.query(
            query_texts=[scene_text],
            n_results=3 # Mengambil 3 dokumen agar konteks visual lebih kaya
        )
        context_list = []
        if results and 'documents' in results and results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                metadata = {}
                # Secara aman mengambil metadata
                if 'metadatas' in results and results['metadatas'] and i < len(results['metadatas'][0]):
                    metadata = results['metadatas'][0][i]
                
                source = metadata.get('image_source', 'Unknown')
                context_list.append({"text": doc, "source": source})
            return context_list
    except Exception as e:
        print(f"RAG Error: {e}")
    return []

def ingest_csv(csv_path: str, batch_size: int = 1000):
    """
    Memproses file CSV dataset gambar (delimiter '|') menjadi embeddings di ChromaDB.
    """
    if not os.path.exists(csv_path):
        print(f"File {csv_path} tidak ditemukan.")
        return
    
    docs = []
    ids = []
    
    import csv
    print(f"Mulai membaca {csv_path}...")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        # Handle jika header ada spasi atau beda case
        for i, row in enumerate(reader):
            caption = row.get('caption')
            if not caption:
                continue
                
            caption = caption.strip()
            if len(caption) > 5:
                docs.append(caption)
                img_name = row.get('Image_name', f'img_{i}').replace('.jpg', '')
                cap_num = row.get('caption_number', '0')
                ids.append(f"{img_name}_cap{cap_num}_{i}")
                
            if len(docs) >= batch_size:
                collection.upsert(documents=docs, ids=ids)
                print(f"Berhasil upsert {batch_size} caption...")
                docs = []
                ids = []
                
        if docs:
            collection.upsert(documents=docs, ids=ids)
            print(f"Berhasil upsert sisa {len(docs)} caption. Ingestion CSV selesai.")

async def get_visual_context(scene_text: str) -> list:
    """
    Retrieves relevant text segments and image sources from ChromaDB based on a query.
    Used to ground the scene generation within the context of the uploaded script.
    """
    print(f"Retrieving visual context for: {scene_text}")
    return await asyncio.to_thread(_query_rag_sync, scene_text)
