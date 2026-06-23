import os
import csv
import json
import shutil
from collections import defaultdict

combined_dir = '/home/firania/Documents/tugasakhir/ai/dataset_combined'
images_its_dir = '/home/firania/Documents/tugasakhir/ai/images_its'
allaboutits_dir = '/home/firania/Documents/tugasakhir/ai/allaboutITS'

caption_csv = '/home/firania/Documents/tugasakhir/ai/caption.csv'
qwen_csv = '/home/firania/Documents/tugasakhir/ai/capt_qwen.csv'
trigger_word = "itstvstyle"

# Clean up existing folder if any
if os.path.exists(combined_dir):
    shutil.rmtree(combined_dir)
os.makedirs(combined_dir, exist_ok=True)

def read_captions(csv_path):
    captions = defaultdict(dict)
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        next(reader, None) # skip header
        for row in reader:
            if len(row) >= 3:
                img_name, idx, text = row[0].strip(), row[1].strip(), row[2].strip()
                # Ensure the key doesn't have multiple extensions
                if not img_name.endswith('.jpg'):
                    img_name += '.jpg'
                captions[img_name][int(idx)] = text
    
    # Combine 5 sentences
    combined_captions = {}
    for img_name, text_dict in captions.items():
        # sort by index
        sorted_texts = [text_dict[k] for k in sorted(text_dict.keys())]
        combined_text = " ".join(sorted_texts)
        combined_captions[img_name] = combined_text
    
    return combined_captions

print("Membaca caption.csv...")
captions_its = read_captions(caption_csv)
print(f"Ditemukan {len(captions_its)} gambar di caption.csv")

print("Membaca capt_qwen.csv...")
captions_allabout = read_captions(qwen_csv)
print(f"Ditemukan {len(captions_allabout)} gambar di capt_qwen.csv")

metadata = []

# Process images_its (1x)
print("Memproses images_its (1x)...")
count_its = 0
for file in os.listdir(images_its_dir):
    if file.endswith(('.jpg', '.png', '.jpeg')):
        if file in captions_its:
            src = os.path.join(images_its_dir, file)
            dst = os.path.join(combined_dir, file)
            shutil.copy2(src, dst)
            
            metadata.append({
                "file_name": file,
                "text": f"{trigger_word}, {captions_its[file]}"
            })
            count_its += 1

print(f"Berhasil memproses {count_its} gambar dari images_its.")

# Process allaboutITS (5x)
print("Memproses allaboutITS (5x augmentation)...")
count_allabout = 0
for file in os.listdir(allaboutits_dir):
    if file.endswith(('.jpg', '.png', '.jpeg')):
        if file in captions_allabout:
            src = os.path.join(allaboutits_dir, file)
            base_name, ext = os.path.splitext(file)
            for i in range(1, 6):
                new_file = f"{base_name}_aug{i}{ext}"
                dst = os.path.join(combined_dir, new_file)
                shutil.copy2(src, dst)
                
                metadata.append({
                    "file_name": new_file,
                    "text": f"{trigger_word}, {captions_allabout[file]}"
                })
            count_allabout += 1

print(f"Berhasil memproses {count_allabout} gambar dari allaboutITS (total {count_allabout * 5} file digenerate).")

# Save metadata.jsonl
print("Menyimpan metadata.jsonl...")
metadata_path = os.path.join(combined_dir, 'metadata.jsonl')
with open(metadata_path, 'w', encoding='utf-8') as f:
    for entry in metadata:
        f.write(json.dumps(entry) + '\n')

print(f"Selesai! {len(metadata)} total entri telah disimpan di {metadata_path}")
