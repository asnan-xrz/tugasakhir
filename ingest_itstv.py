import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import math

def ingest_data(csv_path: str, db_path: str, collection_name: str, batch_size: int = 100):
    """
    Reads data from a CSV file and ingests it into a ChromaDB collection in batches.
    """
    print("=== Storyboard AI Source Grounding Ingestion ===")
    
    if not os.path.exists(csv_path):
        print(f"Error: File CSV tidak ditemukan di '{csv_path}'.")
        return

    # Pastikan folder database ada
    os.makedirs(db_path, exist_ok=True)
        
    print(f"Menghubungkan ke ChromaDB di '{db_path}'...")
    try:
        client = chromadb.PersistentClient(path=db_path)
    except Exception as e:
        print(f"Error saat menghubungkan ke ChromaDB: {e}")
        return

    # --- PENTING: Hapus koleksi lama agar metadata tidak 'nyampur' atau kosong ---
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
    
    print(f"Membaca dataset dari '{csv_path}'...")
    # Separator | sesuai format dataset kamu
    df = pd.read_csv(csv_path, sep='|')
        
    # Sesuai snippet kamu: image_name (huruf kecil)
    required_columns = ['image_name', 'caption_number', 'caption']
    
    # Cek kolom dan lakukan rename otomatis jika ada perbedaan kapitalisasi
    for col in required_columns:
        if col not in df.columns:
            # Cek apakah ada versi Capitalized-nya (Image_name)
            cap_col = col.capitalize()
            if cap_col in df.columns:
                df.rename(columns={cap_col: col}, inplace=True)
            else:
                print(f"Error: Kolom '{col}' tidak ditemukan. Kolom yang ada: {df.columns.tolist()}")
                return
            
    df = df.dropna(subset=['caption'])
    total_rows = len(df)
    
    print(f"Total data valid: {total_rows}")
        
    num_batches = math.ceil(total_rows / batch_size)
    print(f"Memulai proses ingestion ({num_batches} batch)...\n")
    
    with tqdm(total=total_rows, desc="Ingesting data") as pbar:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_rows)
            
            batch_df = df.iloc[start_idx:end_idx]
            
            # ID unik
            ids = [f"id_{idx}" for idx in range(start_idx, end_idx)]
            documents = batch_df['caption'].astype(str).tolist()
            
            # KEY metadata diubah jadi 'image_source' agar sinkron dengan ingest_test.py
            metadatas = [
                {
                    "image_source": str(row['image_name']),
                    "caption_number": int(row['caption_number'])
                }
                for _, row in batch_df.iterrows()
            ]
            
            try:
                collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            except Exception as e:
                print(f"Error pada batch {i+1}: {e}")
                
            pbar.update(len(batch_df))
            
    print("\n✅ Ingestion selesai! Sekarang jalankan ingest_test.py, metadata pasti muncul.")

if __name__ == "__main__":
    # Load token dari config.env jika ada
    if os.path.exists("config.env"):
        from dotenv import load_dotenv
        load_dotenv("config.env")

    # Konfigurasi path untuk Katana-15 kamu
    CSV_FILE_PATH = "ai/caption.csv"
    DB_FOLDER_PATH = "./backend/chroma_db"
    COLLECTION_NAME = "its_tv_scripts"
    
    ingest_data(
        csv_path=CSV_FILE_PATH, 
        db_path=DB_FOLDER_PATH, 
        collection_name=COLLECTION_NAME
    )