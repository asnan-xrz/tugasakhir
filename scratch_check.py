import csv
from collections import defaultdict

input_file = '/home/firania/Documents/tugasakhir/ai/capt_qwen.csv'

rows = []
with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='|')
    header = next(reader)
    for row in reader:
        if row:
            rows.append(row)

grouped = defaultdict(list)
for row in rows:
    img = row[0]
    grouped[img].append(row)

issues = 0
for img, group in grouped.items():
    if len(group) == 5:
        captions = [r[2] for r in group]
        if len(set(captions)) < 5:
            print(f"{img} has duplicated captions. Unique count: {len(set(captions))}")
            issues += 1

print(f"Total issues: {issues}")
