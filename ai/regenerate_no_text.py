import os
import csv
import re
import dashscope

dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

image_dir = '/home/firania/Documents/tugasakhir/ai/allaboutITS'
csv_file = '/home/firania/Documents/tugasakhir/ai/capt_qwen.csv'

prompt = """Deskripsikan gambar terkait Institut Teknologi Sepuluh Nopember (ITS) ini SECARA SANGAT DETAIL ke dalam 5 kalimat yang berbeda dalam bahasa Indonesia.

Instruksi Detail:
1. Kalimat pertama: Deskripsikan subjek utama gambar secara umum (apakah itu sekelompok orang, satu orang, atau sebuah bangunan/pemandangan jika tidak ada orang).
2. Kalimat kedua: Jelaskan secara spesifik detail dari subjek utama tersebut (seperti pakaian/seragam/atribut jika ada orang, atau detail arsitektur/bentuk jika berupa bangunan).
3. Kalimat ketiga: Deskripsikan aktivitas atau suasana utama (apa yang sedang dilakukan subjek, atau bagaimana suasana lingkungan tersebut).
4. Kalimat keempat: Deskripsikan elemen visual pendukung atau teks/logo yang menonjol. JIKA TIDAK ADA TEKS/ANGKA/LOGO, DILARANG KERAS menulis 'Tidak ada teks/angka/logo', cukup deskripsikan benda pendukung lain secara mendetail.
5. Kalimat kelima: Gambarkan latar belakang atau kondisi lingkungan secara keseluruhan (cuaca, langit, pencahayaan, atau setting tempat).

ATURAN WAJIB DAN MUTLAK:
- HANYA outputkan 5 kalimat terpisah secara berurutan.
- DILARANG KERAS menggunakan kalimat negatif seperti 'Tidak ada orang', 'Tidak terlihat', 'Tidak ada teks', 'Tidak ada angka', 'Tidak ada logo', dsb. Jika sesuatu tidak ada, JANGAN disebutkan sama sekali. Fokus hanya pada apa yang ADA di dalam gambar!
- TIDAK BOLEH ada awalan angka (1, 2, 3), bullet point, atau strip (-).
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
        # Gunakan regex untuk memisahkan jika model mengembalikan 1 paragraf panjang
        if '\n' not in text.strip() and '. ' in text:
            pattern = r'(?<!\bProf)(?<!\bDr)(?<!\bIr)(?<!\bDrs)(?<!\bdll)(?<!\bdsb)(?<!\bdkk)(?<!\bdst)(?<!\bno)(?<!\bH)(?<!\bHj)(?<!\b[A-Z])\.\s+'
            lines = [s.strip() for s in re.split(pattern, text.strip()) if s.strip()]
            for i in range(len(lines)):
                if not lines[i].endswith('.'):
                    lines[i] += '.'
        else:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        cleaned_lines = []
        for line in lines:
            line = re.sub(r'^[\*\-\d\.]+\s*', '', line)
            line = line.replace('"', '').replace("'", "")
            if line:
                cleaned_lines.append(line)
                
        if not cleaned_lines:
            cleaned_lines = ["Gambar tidak dapat dideskripsikan."]
        
        # Pad or trim to exactly 5 lines
        if len(cleaned_lines) > 5:
            # Join extra lines to the 5th line
            cleaned_lines[4] = " ".join(cleaned_lines[4:])
            cleaned_lines = cleaned_lines[:5]
        
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
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        header = next(reader)
        for row in reader:
            if len(row) == 3:
                img, idx, cap = row
                rows.append(row)
                cap_lower = cap.lower()
                if 'tidak ada' in cap_lower and ('teks' in cap_lower or 'angka' in cap_lower or 'logo' in cap_lower or 'tulisan' in cap_lower):
                    images_to_regenerate.add(img)

    print(f"Total gambar yang akan diregenerate: {len(images_to_regenerate)}")

    new_captions_dict = {}
    current_img_idx = 0
    total_imgs = len(images_to_regenerate)
    
    for img_name in sorted(list(images_to_regenerate)):
        current_img_idx += 1
        print(f"[{current_img_idx}/{total_imgs}] Meregenerate caption untuk {img_name}...")
        img_path = os.path.join(image_dir, img_name)
        
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
