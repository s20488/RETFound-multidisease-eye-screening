import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


def is_grayscale(image_path):
    image = Image.open(image_path).convert("RGB")
    pixels = list(image.getdata())
    for pixel in pixels:
        if pixel[0] != pixel[1] or pixel[1] != pixel[2]:
            return False
    return True


def process_image(file_path):
    try:
        if is_grayscale(file_path):
            os.remove(file_path)
            print(f"Deleted black and white image: {file_path}")
    except Exception as e:
        print(f"Error in processing of {file_path}: {e}")


def remove_grayscale_images_parallel(directory):
    with ThreadPoolExecutor() as executor:
        file_paths = [os.path.join(directory, f) for f in os.listdir(directory)]
        executor.map(process_image, file_paths)


image_directory = "/mnt/data/cfi"
remove_grayscale_images_parallel(image_directory)
