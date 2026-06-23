import csv

input_file = '/home/firania/Documents/tugasakhir/ai/capt_qwen.csv'

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith('Q180.jpg|3|'):
        new_lines.append('Q180.jpg|3|Monumen tersebut memiliki tulisan HARI INI ADA, TANGGAL 15 NOVEMBER 1977 dan nama Prof. Dr. Sjarif Thaib.\n')
    elif line.startswith('Q180.jpg|4|'):
        new_lines.append('Q180.jpg|4|Latar belakangnya adalah area kampus ITS dengan bangunan modern dan taman kecil di sekitarnya.\n')
    elif line.startswith('Q311.jpg|3|'):
        new_lines.append('Q311.jpg|3|Di belakang mereka terdapat spanduk bertuliskan PROF. HEDAYAT HARDDIO MASIRAN beserta tanggal 17 Oktober 2019 dan nama institusi ITS.\n')
    elif line.startswith('Q311.jpg|4|'):
        new_lines.append('Q311.jpg|4|Latar belakangnya adalah panggung dengan dinding berpola geometris dan beberapa tanaman hias di depan panggung.\n')
    else:
        new_lines.append(line)

with open(input_file, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Patch applied.")
