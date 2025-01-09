import os
import shutil

base_dir = "/mnt/data/cfi"

image_extensions = {".dcm"}

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if any(file.lower().endswith(ext) for ext in image_extensions):
            file_path = os.path.join(root, file)

            target_path = os.path.join(base_dir, file)

            try:
                shutil.move(file_path, target_path)
                print(f"Moved: {file_path} -> {target_path}")
            except Exception as e:
                print(f"Error moving {file_path}: {e}")
