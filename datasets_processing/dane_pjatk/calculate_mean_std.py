from PIL import Image
import numpy as np
import os


def calculate_mean_std(image_folder):
    pixel_values = []

    for image_file in os.listdir(image_folder):
        if image_file.endswith(".png"):
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
            image_array = np.array(image) / 255.0
            pixel_values.append(image_array)

    pixel_values = np.concatenate([img.reshape(-1, 3) for img in pixel_values], axis=0)

    mean = np.mean(pixel_values, axis=0)
    std = np.std(pixel_values, axis=0)

    return mean, std


image_folder = "/mnt/data/cfi"

mean, std = calculate_mean_std(image_folder)
print(f"Mean: {mean}, Std: {std}")
