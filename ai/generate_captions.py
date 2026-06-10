import os
import base64
import requests
import csv
import re
import sys
from PIL import Image
import io

image_dir = '/home/firania/Documents/tugasakhir/ai/allaboutITS'
output_csv = '/home/firania/Documents/tugasakhir/ai/captionITS.csv'

# Prompt untuk llava
prompt = """Deskripsikan gambar-gambar Institut Teknologi Sepuluh Nopember ini secara sangat detail ke dalam 5 kalimat yang berbeda dalam bahasa Indonesia.
Fokuslah pada elemen-elemen berikut:
- Teks atau tulisan yang terlihat di dalam gambar.
- Warna-warna dominan pada objek.
- Jumlah orang (jika ada).
- Benda atau objek utama yang ada di dalam gambar.  (Mahasiswa, Rektor, Orang tua Mahasiswa)
- Suasana atau lokasi.

Berikan HANYA 5 kalimat terpisah. Pisahkan masing-masing kalimat dengan baris baru (Enter). 
JANGAN menambahkan nomor urut. JANGAN menambahkan kata pengantar atau penutup. Langsung 5 kalimat deskripsi. Sertakan kata-kata Institut Teknologi Sepuluh Nopember karena banyak logo """

def encode_image(image_path):
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Resize maksimal 512x512 agar hemat token dan memori
        img.thumbnail((512, 512))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def get_captions(image_path):
    base64_image = encode_image(image_path)
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llava",
        "prompt": prompt,
        "images": [base64_image],
        "stream": False,
        "options": {
            "temperature": 0.4
        }
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        text = result.get('response', '')
        
        # Bersihkan dan ambil kalimat
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        cleaned_lines = []
        for line in lines:
            # Hapus angka di awal kalimat jika llava tetap memberikan nomor (misal "1. ", "1- ", dll)
            line = re.sub(r'^[\*\-\d\.]+\s*', '', line)
            # Hapus quote jika ada
            line = line.replace('"', '').replace("'", "")
            if line:
                cleaned_lines.append(line)
                
        # Pastikan ada 5 kalimat, jika kurang duplikasi yang terakhir atau beri fallback
        if not cleaned_lines:
            cleaned_lines = ["Gambar tidak dapat dideskripsikan."]
        
        while len(cleaned_lines) < 5:
            cleaned_lines.append(cleaned_lines[-1])
            
        return cleaned_lines[:5]
    except Exception as e:
        print(f"Error memproses {image_path}: {e}")
        return ["Gagal menghasilkan deskripsi"] * 5

images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
total = len(images)

if total == 0:
    print("Tidak ada file .jpg ditemukan di", image_dir)
    sys.exit()

# Cek gambar yang sudah diproses agar bisa lanjut (resume)
processed_images = set()
if os.path.exists(output_csv):
    with open(output_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        next(reader, None) # skip header
        for row in reader:
            if row:
                processed_images.add(row[0])

images_to_process = [img for img in images if img not in processed_images]

if not images_to_process:
    print("Semua gambar sudah selesai diproses!")
    sys.exit()

print(f"Memulai proses captioning untuk {len(images_to_process)} gambar sisa (dari total {total} gambar) menggunakan model llava lokal...")

mode = 'a' if os.path.exists(output_csv) else 'w'
with open(output_csv, mode, newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='|')
    if mode == 'w':
        writer.writerow(['Image_name', 'caption_number', 'caption'])
    
    for i, img_name in enumerate(images_to_process):
        print(f"[{i+1}/{len(images_to_process)}] Memproses {img_name}...")
        img_path = os.path.join(image_dir, img_name)
        captions = get_captions(img_path)
        
        for idx, cap in enumerate(captions):
            writer.writerow([img_name, idx, cap])
        
        # Flush agar data langsung tersimpan ke disk per gambar
        f.flush()

print(f"Selesai! Hasil telah disimpan ke {output_csv}")
