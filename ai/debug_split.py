import csv
import re

input_file = '/home/firania/Documents/tugasakhir/ai/capt_qwen.csv'

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='|')
    header = next(reader)
    rows = list(reader)

from collections import defaultdict
image_groups = defaultdict(list)
for r in rows:
    if len(r) == 3:
        image_groups[r[0]].append(r)

for img, group in image_groups.items():
    if len(group) == 5:
        captions = [r[2] for r in group]
        if all(c == captions[0] for c in captions) and len(captions[0].strip()) > 0 and '. ' in captions[0]:
            print(f"Image {img} is duplicated!")
            pattern = r'(?<!\bProf)(?<!\bDr)(?<!\bIr)(?<!\bDrs)(?<!\bdll)(?<!\bdsb)(?<!\bdkk)(?<!\bdst)(?<!\bno)(?<!\bH)(?<!\bHj)(?<!\b[A-Z])\.\s+'
            sentences = [s.strip() for s in re.split(pattern, captions[0]) if s.strip()]
            print(f"Sentences count: {len(sentences)}")
