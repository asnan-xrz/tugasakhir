import os
import glob

directory = "/home/firania/Documents/tugasakhir/ai/allaboutITS"
files = sorted(os.listdir(directory))

image_exts = ('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp')

count = 1
for f in files:
    if f.lower().endswith(image_exts):
        old_path = os.path.join(directory, f)
        ext = os.path.splitext(f)[1]
        new_name = f"Q{count:03d}{ext}"
        new_path = os.path.join(directory, new_name)
        
        # Avoid renaming if the name is already the target (though we just started Q)
        if old_path != new_path:
            # Handle potential conflicts if a file already exists with the new name
            # Since we sort first, it should be fine as long as there are no existing Qxxx files
            os.rename(old_path, new_path)
        count += 1

print(f"Renamed {count - 1} image files.")
