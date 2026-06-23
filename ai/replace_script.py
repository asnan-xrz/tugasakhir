import re

file_path = '/home/firania/Documents/tugasakhir/ai/capt_qwen.csv'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace Boneka -> Boneka (bernama Cak Seno, sebagai maskot ITS)
# Keep the original casing of the first letter if possible, or just replace with exact case match.
def replace_boneka(match):
    word = match.group(0)
    return f"{word} (bernama Cak Seno, sebagai maskot ITS)"

def replace_karakter(match):
    word = match.group(0)
    # Ignore if it's part of a word or already replaced
    return f"{word} (bernama Cak Seno, sebagai maskot ITS)"

# To prevent double replacing, first remove any existing (bernama Cak Seno, sebagai maskot ITS)
content = content.replace(" (bernama Cak Seno, sebagai maskot ITS)", "")

content = re.sub(r'\b(Boneka|boneka)\b', replace_boneka, content)
content = re.sub(r'\b(Karakter|karakter)\b', replace_karakter, content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Done")
