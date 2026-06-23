import csv
import re

file_path = '/home/firania/Documents/tugasakhir/ai/capt_qwen.csv'
images_to_regenerate = set()

pattern = re.compile(r'tidak\s+(ada|terdapat|terlihat|menampilkan).*?(teks|angka|logo|tulisan)', re.IGNORECASE)

with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='|')
    next(reader)
    for row in reader:
        if len(row) == 3:
            img, idx, cap = row
            if pattern.search(cap):
                images_to_regenerate.add(img)

print(f"Total images to regenerate: {len(images_to_regenerate)}")
print("Images:", sorted(list(images_to_regenerate)))
