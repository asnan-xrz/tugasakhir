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
    # check if we have 5 rows and all captions are identical
    if len(group) == 5:
        captions = [r[2] for r in group]
        if all(c == captions[0] for c in captions) and len(captions[0].strip()) > 0:
            # We need to split captions[0] into 5 sentences
            text = captions[0]
            # Split by period followed by space
            sentences = [s.strip() for s in re.split(r'\.\s+', text) if s.strip()]
            
            # Re-add period if it was removed by split
            for i in range(len(sentences)):
                if not sentences[i].endswith('.'):
                    sentences[i] += '.'
                    
            if len(sentences) == 5:
                for i in range(5):
                    fixed_rows.append([img, str(i), sentences[i]])
                continue
            elif len(sentences) > 5:
                # If more than 5, join the extra ones to the last sentence
                for i in range(4):
                    fixed_rows.append([img, str(i), sentences[i]])
                last_sentence = " ".join(sentences[4:])
                fixed_rows.append([img, '4', last_sentence])
                continue
            elif len(sentences) > 0:
                # If fewer than 5 sentences but more than 0, pad with empty or just use what we have
                # Actually let's just see how many such cases exist.
                print(f"Warning: {img} has {len(sentences)} sentences.")
                for i in range(5):
                    if i < len(sentences):
                        fixed_rows.append([img, str(i), sentences[i]])
                    else:
                        fixed_rows.append([img, str(i), sentences[-1]])
                continue

    # if not identical or not 5 rows, just keep as is
    for r in group:
        fixed_rows.append(r)

with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, delimiter='|')
    writer.writerow(header)
    writer.writerows(fixed_rows)

print("Done. Fixed CSV saved to capt_qwen_fixed.csv")
