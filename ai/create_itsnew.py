import nbformat as nbf

nb = nbf.v4.new_notebook()

cell1 = nbf.v4.new_code_cell("""import torch
import subprocess

def check_gpu_status():
    print("--- Pengecekan via PyTorch ---")
    cuda_tersedia = torch.cuda.is_available()
    print(f"CUDA Tersedia: {cuda_tersedia}")
    
    if cuda_tersedia:
        gpu_count = torch.cuda.device_count()
        print(f"Jumlah GPU: {gpu_count}")
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"VRAM Total: {mem_total:.2f} GB")
    else:
        print("Peringatan: CUDA tidak terdeteksi oleh PyTorch.")

    print("\\n--- Pengecekan via Sistem (nvidia-smi) ---")
    try:
        nvidia_smi = subprocess.check_output("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader", shell=True)
        print(f"Driver Info: {nvidia_smi.decode('utf-8').strip()}")
    except Exception:
        print("Gagal menjalankan nvidia-smi.")

if __name__ == "__main__":
    check_gpu_status()""")

cell2 = nbf.v4.new_code_cell("""# Install dependensi utama
!pip install -q diffusers accelerate transformers bitsandbytes xformers ftfy pandas matplotlib seaborn tensorboard datasets peft huggingface_hub torchvision
!pip install -q -U git+https://github.com/huggingface/diffusers""")

cell3 = nbf.v4.new_code_cell("""import os
os.chdir('/home/firania/Documents/tugasakhir/ai')

# Download script training LoRA dari huggingface diffusers v0.37.0
if os.path.exists("train_text_to_image_lora.py"):
    os.remove("train_text_to_image_lora.py")

!wget https://raw.githubusercontent.com/huggingface/diffusers/v0.37.0/examples/text_to_image/train_text_to_image_lora.py

if os.path.exists("train_text_to_image_lora.py"):
    print("✅ File train_text_to_image_lora.py berhasil didownload!")""")

cell4 = nbf.v4.new_code_cell("""# Persiapan folder output
import os
OUTPUT_DIR = "/home/firania/Documents/tugasakhir/ai/lora_output_its"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Folder output disiapkan di: {OUTPUT_DIR}")""")

cell5 = nbf.v4.new_code_cell("""# JALANKAN TRAINING LORA
# Pastikan tidak ada spasi yang salah di argumen
!accelerate launch train_text_to_image_lora.py \\
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\
  --train_data_dir="/home/firania/Documents/tugasakhir/ai/dataset_combined" \\
  --dataloader_num_workers=4 \\
  --resolution=512 \\
  --center_crop \\
  --train_batch_size=1 \\
  --gradient_accumulation_steps=4 \\
  --max_train_steps=5000 \\
  --learning_rate=1e-04 \\
  --max_grad_norm=1 \\
  --lr_scheduler="cosine" \\
  --lr_warmup_steps=0 \\
  --output_dir="/home/firania/Documents/tugasakhir/ai/lora_output_its" \\
  --checkpointing_steps=500 \\
  --validation_prompt="itstvstyle, a photo of Cak Seno mascot ITS standing in front of rectorate building" \\
  --seed=1337""")

nb.cells = [
    nbf.v4.new_markdown_cell("# 1. Persiapan Environment & GPU"), cell1,
    nbf.v4.new_markdown_cell("# 2. Install Dependencies"), cell2,
    nbf.v4.new_markdown_cell("# 3. Persiapan Script Training (Diffusers)"), cell3,
    nbf.v4.new_markdown_cell("# 4. Setup Output Directory"), cell4,
    nbf.v4.new_markdown_cell("# 5. Eksekusi Training Model Diffusion (LoRA)\\n**Perhatian**: `--random_flip` dimatikan jika tidak ingin merusak teks/logo. Jika dataset sudah diatur duplikat fisik, jangan pakai flag `--random_flip`."), cell5
]

with open('/home/firania/Documents/tugasakhir/ai/itsnew.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook itsnew.ipynb updated successfully.")
