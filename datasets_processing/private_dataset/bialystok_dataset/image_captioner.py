import os
from PIL import Image, ImageDraw, ImageFont
import random

base_path = "/mnt/data/cfi_manual_cataract/train"
cataract_dir = os.path.join(base_path, "cataract")
normal_dir = os.path.join(base_path, "normal")

FONT_SIZE = 20
TEXT_HEIGHT = 30
FONT_COLOR = (255, 255, 255)
BACKGROUND_COLOR = (0, 0, 0)


def get_random_images(folder, num=5):
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(files)
    return [os.path.join(folder, f) for f in files[:num]]


def add_caption(img, text):
    new_height = img.height + TEXT_HEIGHT
    new_img = Image.new('RGB', (img.width, new_height), color=BACKGROUND_COLOR)
    new_img.paste(img, (0, 0))

    draw = ImageDraw.Draw(new_img)

    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except:
        font = ImageFont.load_default()

    text_width = draw.textlength(text, font=font)
    x = (img.width - text_width) // 2
    y = img.height + (TEXT_HEIGHT - FONT_SIZE) // 2

    draw.text((x, y), text, font=font, fill=FONT_COLOR)
    return new_img


def process_images(image_paths, label, resize_to=None):
    images = []
    for path in image_paths:
        img = Image.open(path)
        if resize_to:
            img = img.resize(resize_to)
        labeled_img = add_caption(img, label)
        images.append(labeled_img)
    return images


cataract_images = get_random_images(cataract_dir)
normal_images = get_random_images(normal_dir)

sample_image = Image.open(cataract_images[0])
img_width, img_height = sample_image.size

c_imgs = process_images(cataract_images, "Cataract", (img_width, img_height))
n_imgs = process_images(normal_images, "Normal", (img_width, img_height))

block_width = img_width
block_height = img_height + TEXT_HEIGHT

combined_image = Image.new(
    'RGB',
    (block_width * 5, block_height * 2)
)

for i, img in enumerate(c_imgs):
    combined_image.paste(img, (i * block_width, 0))

for i, img in enumerate(n_imgs):
    combined_image.paste(img, (i * block_width, block_height))

combined_image.save("combined_with_labels.jpg")
