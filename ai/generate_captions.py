import os
import base64
import requests
import csv
import re
import sys
from PIL import Image
import io
import dashscope

dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

image_dir = '/home/firania/Documents/tugasakhir/ai/allaboutITS'
output_csv = '/home/firania/Documents/tugasakhir/ai/capt_qwen.csv'

# Prompt untuk Qwen2.5-VL
prompt = """Deskripsikan gambar terkait Institut Teknologi Sepuluh Nopember (ITS) ini SECARA SANGAT DETAIL ke dalam 5 kalimat yang berbeda dalam bahasa Indonesia.

Instruksi Detail:
1. Kalimat pertama: Wajib sebutkan jumlah orang yang terlihat (misal: satu, dua, banyak) beserta jenis kelaminnya (laki-laki/perempuan) atau profesinya.
2. Kalimat kedua: Jelaskan secara spesifik pakaian/seragam dan atribut yang mereka kenakan (beserta warnanya).
3. Kalimat ketiga: Deskripsikan aksi, ekspresi, atau aktivitas yang sedang mereka lakukan di gambar tersebut.
4. Kalimat keempat: Baca dan tuliskan dengan jelas jika ada teks, angka, atau tulisan pada spanduk, logo, maupun objek lainnya.
5. Kalimat kelima: Gambarkan latar belakang tempatnya (contoh: di dalam kelas, di lahan parkir, taman, dll) serta benda-benda pendukung di sekitarnya.

ATURAN WAJIB:
- HANYA outputkan 5 kalimat terpisah (harus merepresentasikan 5 poin di atas secara berurutan).
- TIDAK BOLEH ada awalan angka (1, 2, 3), bullet point, atau strip (-).
- TIDAK BOLEH ada teks pengantar seperti 'Berikut adalah deskripsinya...'.
- Setiap kalimat harus diakhiri dengan tanda titik dan enter."""

def get_captions(image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"file://{os.path.abspath(image_path)}"},
                {"text": prompt}
            ]
        }
    ]
    try:
        response = dashscope.MultiModalConversation.call(
            api_key="sk-ws-H.IYHRDX.dwdU.MEYCIQDzEIzROjbd73EaHeb1HGpxTIekn2JLdaTXlwmEjUASLQIhANIvKDU9kURrm6vX83c3wnAPlEc-hI8MTiPgJhv0AYmI",
            model='qwen-vl-plus',
            messages=messages
        )
        
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            text = content[0]["text"] if isinstance(content, list) else content
        else:
            print(f"API Error: {response.code} - {response.message}")
            return ["Gagal menghasilkan deskripsi"] * 5
        
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

print(f"Memulai proses captioning untuk {len(images_to_process)} gambar sisa (dari total {total} gambar) menggunakan model qwen3.7-plus...")

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
