import os
import csv
import re
import dashscope

dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

image_dir = '/home/firania/Documents/tugasakhir/ai/allaboutITS'
csv_file = '/home/firania/Documents/tugasakhir/ai/capt_qwen.csv'

prompt = """Deskripsikan gambar terkait Institut Teknologi Sepuluh Nopember (ITS) ini SECARA SANGAT DETAIL ke dalam 5 kalimat yang berbeda dalam bahasa Indonesia. Gambar ini berfokus pada pemandangan, bangunan, objek, atau lingkungan tanpa kehadiran manusia secara dominan.

Instruksi Detail:
1. Kalimat pertama: Deskripsikan subjek atau objek utama dalam gambar secara keseluruhan (misalnya bangunan gedung, taman, spanduk, patung, dll).
2. Kalimat kedua: Jelaskan secara spesifik detail arsitektur, bentuk, warna, atau material dari objek utama tersebut.
3. Kalimat ketiga: Baca dan tuliskan dengan jelas jika ada teks, angka, atau tulisan pada spanduk, logo, maupun objek lainnya yang menonjol. Jika tidak ada, deskripsikan elemen visual utama lainnya.
4. Kalimat keempat: Deskripsikan elemen pendukung di sekitar objek utama (misalnya tanaman, pohon, kendaraan yang terparkir, tiang, dll).
5. Kalimat kelima: Gambarkan suasana lingkungan secara keseluruhan (langit, awan, pencahayaan, atau kesan umum dari tempat tersebut).

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
            line = re.sub(r'^[\*\-\d\.]+\s*', '', line)
            line = line.replace('"', '').replace("'", "")
            if line:
                cleaned_lines.append(line)
                
        if not cleaned_lines:
            cleaned_lines = ["Gambar tidak dapat dideskripsikan."]
        
        while len(cleaned_lines) < 5:
            cleaned_lines.append(cleaned_lines[-1])
            
        return cleaned_lines[:5]
    except Exception as e:
        print(f"Error memproses {image_path}: {e}")
        return ["Gagal menghasilkan deskripsi"] * 5

def main():
    images_to_regenerate = set()
    rows = []
    header = []
    
    # Baca file CSV
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        header = next(reader)
        for row in reader:
            if len(row) == 3:
                img, idx, cap = row
                rows.append(row)
                cap_lower = cap.lower()
                if ('tidak ada' in cap_lower or 'tidak terlihat' in cap_lower) and ('orang' in cap_lower or 'manusia' in cap_lower or 'laki-laki' in cap_lower or 'perempuan' in cap_lower or 'individu' in cap_lower):
                    images_to_regenerate.add(img)

    print(f"Total gambar yang akan diregenerate: {len(images_to_regenerate)}")

    # Buat dictionary untuk menyimpan row yang sudah diregenerate
    new_rows = []
    current_img_idx = 0
    total_imgs = len(images_to_regenerate)
    
    images_to_regenerate = sorted(list(images_to_regenerate))
    
    new_captions_dict = {}
    
    for img_name in images_to_regenerate:
        current_img_idx += 1
        print(f"[{current_img_idx}/{total_imgs}] Meregenerate caption untuk {img_name}...")
        img_path = os.path.join(image_dir, img_name)
        
        captions = get_captions(img_path)
        new_captions_dict[img_name] = captions

    # Update rows dengan caption baru
    for i in range(len(rows)):
        img = rows[i][0]
        idx = int(rows[i][1])
        if img in new_captions_dict:
            # Pastikan idx <= 4 (karena index caption 0-4)
            if idx < len(new_captions_dict[img]):
                rows[i][2] = new_captions_dict[img][idx]

    # Simpan kembali ke file CSV
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow(header)
        writer.writerows(rows)
        
    print(f"Selesai! File {csv_file} telah diperbarui.")

if __name__ == '__main__':
    main()
