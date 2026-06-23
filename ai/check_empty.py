import csv

file_path = '/home/firania/Documents/tugasakhir/ai/capt_qwen.csv'
images_to_regenerate = set()

with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='|')
    next(reader)
    for row in reader:
        if len(row) == 3:
            img, idx, cap = row
            # Any sentence that strongly implies there are no people
            cap_lower = cap.lower()
            if ('tidak ada' in cap_lower or 'tidak terlihat' in cap_lower) and ('orang' in cap_lower or 'manusia' in cap_lower or 'laki-laki' in cap_lower or 'perempuan' in cap_lower or 'individu' in cap_lower):
                images_to_regenerate.add(img)

print(f"Total images to regenerate: {len(images_to_regenerate)}")
