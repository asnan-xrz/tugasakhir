import csv

file_path = '/home/firania/Documents/tugasakhir/ai/capt_qwen.csv'
images_to_regenerate = set()

with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='|')
    next(reader)
    for row in reader:
        if len(row) == 3:
            img, idx, cap = row
            cap_lower = cap.lower()
            if 'tidak ada' in cap_lower and ('teks' in cap_lower or 'angka' in cap_lower or 'logo' in cap_lower or 'tulisan' in cap_lower):
                images_to_regenerate.add(img)

print(f"Total images to regenerate: {len(images_to_regenerate)}")
