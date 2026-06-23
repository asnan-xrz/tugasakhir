import os
import csv
import re
import dashscope

dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

image_dir = '/home/firania/Documents/tugasakhir/ai/allaboutITS'
csv_file = '/home/firania/Documents/tugasakhir/ai/capt_qwen.csv'

# Prompt for when there's "tidak ada" or "gagal"
prompt = """Deskripsikan gambar terkait Institut Teknologi Sepuluh Nopember (ITS) ini SECARA SANGAT DETAIL ke dalam 5 kalimat yang berbeda dalam bahasa Indonesia. Gambar ini mungkin berfokus pada pemandangan, bangunan, objek, atau lingkungan tanpa kehadiran manusia secara dominan, atau mungkin tidak ada teks yang menonjol.

Instruksi Detail:
1. Kalimat pertama: Deskripsikan subjek atau objek utama dalam gambar secara keseluruhan (misalnya bangunan gedung, taman, spanduk, patung, orang, dll).
2. Kalimat kedua: Jelaskan secara spesifik detail arsitektur, bentuk, warna, pakaian, atau material dari objek/subjek utama tersebut.
3. Kalimat ketiga: Deskripsikan aksi, ekspresi, atau aktivitas yang sedang dilakukan jika ada orang, ATAU deskripsikan elemen visual utama lainnya jika tidak ada orang.
4. Kalimat keempat: Baca dan tuliskan dengan jelas jika ada teks, angka, atau tulisan. Jika tidak ada teks, deskripsikan elemen pendukung di sekitar objek utama (misalnya tanaman, pohon, kendaraan, dll).
5. Kalimat kelima: Gambarkan suasana lingkungan secara keseluruhan (langit, awan, pencahayaan, atau kesan umum dari tempat tersebut).

ATURAN WAJIB:
- HANYA outputkan 5 kalimat terpisah (harus merepresentasikan 5 poin di atas secara berurutan).
- JANGAN PERNAH menggunakan kata-kata "Tidak ada", "Tidak terlihat", atau "Tidak terdapat". Jika sesuatu tidak ada, deskripsikan hal lain yang ADA di gambar.
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
    images_to_regenerate.add('Q260.jpg') # User explicitly asked for this
    rows = []
    header = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        header = next(reader)
        for row in reader:
            if len(row) == 3:
                img, idx, cap = row
                rows.append(row)
                cap_lower = cap.lower()
                if 'tidak terdapat' in cap_lower or 'tidak ada' in cap_lower or 'tidak terlihat' in cap_lower or 'gagal menghasilkan' in cap_lower:
                    images_to_regenerate.add(img)

    print(f"Total gambar yang akan diregenerate: {len(images_to_regenerate)}")

    new_rows = []
    current_img_idx = 0
    total_imgs = len(images_to_regenerate)
    
    images_to_regenerate = sorted(list(images_to_regenerate))
    
    new_captions_dict = {}
    
    for img_name in images_to_regenerate:
        current_img_idx += 1
        print(f"[{current_img_idx}/{total_imgs}] Meregenerate caption untuk {img_name}...")
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Gambar {img_path} tidak ditemukan, skip.")
            continue
        
        captions = get_captions(img_path)
        new_captions_dict[img_name] = captions

    for i in range(len(rows)):
        img = rows[i][0]
        idx = int(rows[i][1])
        if img in new_captions_dict:
            if idx < len(new_captions_dict[img]):
                rows[i][2] = new_captions_dict[img][idx]

    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerow(header)
        writer.writerows(rows)
        
    print(f"Selesai! File {csv_file} telah diperbarui.")

if __name__ == '__main__':
    main()
