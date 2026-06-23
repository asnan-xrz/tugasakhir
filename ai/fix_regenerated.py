import csv
import re

input_file = '/home/firania/Documents/tugasakhir/ai/capt_qwen.csv'
output_file = '/home/firania/Documents/tugasakhir/ai/capt_qwen_fixed.csv'

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='|')
    header = next(reader)
    rows = list(reader)

from collections import defaultdict
image_groups = defaultdict(list)
for r in rows:
    if len(r) == 3:
        image_groups[r[0]].append(r)

fixed_rows = []
for img, group in image_groups.items():
    if len(group) == 5:
        captions = [r[2] for r in group]
        # If all captions are identical AND it looks like it contains multiple sentences
        if all(c == captions[0] for c in captions) and len(captions[0].strip()) > 0 and '. ' in captions[0]:
            text = captions[0]
            # split by period followed by space, but avoid common abbreviations
            pattern = r'(?<!\bProf)(?<!\bDr)(?<!\bIr)(?<!\bDrs)(?<!\bdll)(?<!\bdsb)(?<!\bdkk)(?<!\bdst)(?<!\bno)(?<!\bH)(?<!\bHj)(?<!\b[A-Z])\.\s+'
            sentences = [s.strip() for s in re.split(pattern, text) if s.strip()]
            
            for i in range(len(sentences)):
                if not sentences[i].endswith('.'):
                    sentences[i] += '.'
                    
            if len(sentences) == 5:
                for i in range(5):
                    fixed_rows.append([img, str(i), sentences[i]])
                continue
            elif len(sentences) > 5:
                for i in range(4):
                    fixed_rows.append([img, str(i), sentences[i]])
                last_sentence = " ".join(sentences[4:])
                fixed_rows.append([img, '4', last_sentence])
                continue
            elif len(sentences) > 0:
                for i in range(5):
                    if i < len(sentences):
                        fixed_rows.append([img, str(i), sentences[i]])
                    else:
                        fixed_rows.append([img, str(i), sentences[-1]])
                continue

    for r in group:
        fixed_rows.append(r)

with open(input_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='|')
    writer.writerow(header)
    writer.writerows(fixed_rows)

print("Split fixed.")
