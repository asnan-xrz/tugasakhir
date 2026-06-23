import os
import json
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import math

def ingest_data(jsonl_path: str, db_path: str, collection_name: str, batch_size: int = 100):
    print("=== Storyboard AI Source Grounding Ingestion (dataset_combined) ===")
    
    if not os.path.exists(jsonl_path):
        print(f"Error: File JSONL tidak ditemukan di '{jsonl_path}'.")
        return

    os.makedirs(db_path, exist_ok=True)
        
    print(f"Menghubungkan ke ChromaDB di '{db_path}'...")
    try:
        client = chromadb.PersistentClient(path=db_path)
    except Exception as e:
        print(f"Error saat menghubungkan ke ChromaDB: {e}")
        return

    # Kita menghapus koleksi lama karena dataset_combined sudah merangkum semuanya (images_its + allaboutits)
    try:
        client.delete_collection(name=collection_name)
        print(f"Koleksi lama '{collection_name}' dihapus untuk pembersihan data baru.")
    except:
        pass
        
    print("Mempersiapkan embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print(f"Membuat koleksi baru '{collection_name}'...")
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef
    )
    
    print(f"Membaca dataset dari '{jsonl_path}'...")
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                
    total_rows = len(data)
    print(f"Total data valid: {total_rows}")
        
    num_batches = math.ceil(total_rows / batch_size)
    print(f"Memulai proses ingestion ({num_batches} batch)...\n")
    
    with tqdm(total=total_rows, desc="Ingesting data") as pbar:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_rows)
            
            batch_data = data[start_idx:end_idx]
            
            # ID unik
            ids = [f"combined_id_{idx}" for idx in range(start_idx, end_idx)]
            documents = [item['text'] for item in batch_data]
            
            # KEY metadata: 'image_source' agar sinkron dengan backend
            metadatas = [
                {
                    "image_source": item['file_name'],
                    "caption_number": 0  # Default value as combined captions don't have multiple numbers anymore
                }
                for item in batch_data
            ]
            
            try:
                collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            except Exception as e:
                print(f"Error pada batch {i+1}: {e}")
                
            pbar.update(len(batch_data))
            
    print("\n✅ Ingestion dataset_combined selesai!")

if __name__ == "__main__":
    if os.path.exists("../config.env"):
        from dotenv import load_dotenv
        load_dotenv("../config.env")
    elif os.path.exists("config.env"):
        from dotenv import load_dotenv
        load_dotenv("config.env")

    JSONL_FILE_PATH = "ai/dataset_combined/metadata.jsonl"
    DB_FOLDER_PATH = "./backend/chroma_db"
    COLLECTION_NAME = "its_tv_scripts"
    
    # Run relative to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    jsonl_abs = os.path.join(base_dir, JSONL_FILE_PATH)
    db_abs = os.path.join(base_dir, DB_FOLDER_PATH)

    ingest_data(
        jsonl_path=jsonl_abs, 
        db_path=db_abs, 
        collection_name=COLLECTION_NAME
    )
