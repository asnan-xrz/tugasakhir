import chromadb
import os
from chromadb.utils import embedding_functions

# 0. HF Token baby

# 1. Koneksi ke database (Pastikan path folder backend benar)
chroma_client = chromadb.PersistentClient(path="./backend/chroma_db")

# 2. Inisialisasi model embedding
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# 3. Akses koleksi yang sudah di-ingest
collection = chroma_client.get_or_create_collection(name="its_tv_scripts", embedding_function=emb_fn)

# 4. Melakukan Query
query = "suasana mahasiswa berjalan di depan gedung rektorat"
results = collection.query(query_texts=[query], n_results=3)

print(f"\n=== Hasil Query Grounding ===")
print(f"Query: {query}\n")

# 5. Iterasi hasil dengan pengecekan aman
if results and results.get('documents') and len(results['documents'][0]) > 0:
    for i in range(len(results['documents'][0])):
        document = results['documents'][0][i]
        
        # Ambil metadata secara aman
        metadata = None
        if results.get('metadatas') and i < len(results['metadatas'][0]):
            metadata = results['metadatas'][0][i]
        
        # Ambil nama file sumber, gunakan default jika tidak ada
        source = metadata.get('image_source', 'Sumber tidak diketahui') if metadata else 'Tanpa Metadata'
        
        print(f"[{i+1}] Deskripsi: {document}")
        print(f"    Referensi File: {source}\n")
else:
    print("Tidak ada hasil yang ditemukan dalam database.")