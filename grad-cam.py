import torch
import models_vit
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os

# Загрузка модели
model = models_vit.__dict__['vit_large_patch16'](
    num_classes=2,
    drop_path_rate=0.2,
    global_pool=True,
)

# Загрузка весов
checkpoint = torch.load('RETFound_cfp_weights.pth', map_location='cpu')
checkpoint_model = checkpoint['model']
state_dict = model.state_dict()
for k in ['head.weight', 'head.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

# Интерполяция позиционных эмбеддингов
interpolate_pos_embed(model, checkpoint_model)

# Загрузка предобученной модели
msg = model.load_state_dict(checkpoint_model, strict=False)

assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

# Инициализация fc слоя
trunc_normal_(model.head.weight, std=2e-5)

print("Model = %s" % str(model))


# Функция для Grad-CAM
def grad_cam_vit(model, img, target_class):
    model.eval()

    # Получаем активации и градиенты последнего блока
    activations = None
    gradients = None

    # Хук для захвата активаций
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output
        activations.retain_grad()  # Важно: сохраняем градиенты для активаций

    # Хук для захвата градиентов
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    # Регистрируем хуки
    hook_forward = model.blocks[-1].register_forward_hook(forward_hook)
    hook_backward = model.blocks[-1].register_backward_hook(backward_hook)

    # Прямой проход
    output = model(img)

    # Обратный проход
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0][target_class] = 1.0
    output.backward(gradient=one_hot)

    # Убедимся, что активации и градиенты получены
    assert activations is not None and gradients is not None, "Градиенты не были вычислены!"

    # Усредняем градиенты по патчам (исключаем cls token)
    weights = torch.mean(gradients[:, 1:], dim=1)  # [batch, num_patches]

    # Собираем карту Grad-CAM
    grads_cam = torch.einsum('bn,bnd->bd', weights, activations[:, 1:])
    grads_cam = F.relu(grads_cam)

    # Ресайз и нормализация
    grads_cam = grads_cam.reshape(-1, 14, 14)  # Для patch_size=16 (224/16=14)
    grads_cam = grads_cam.detach().cpu().numpy()
    grads_cam = cv2.resize(grads_cam, (img.shape[2], img.shape[3]))
    grads_cam = (grads_cam - grads_cam.min()) / (grads_cam.max() - grads_cam.min() + 1e-8)

    # Удаляем хуки
    hook_forward.remove()
    hook_backward.remove()

    return grads_cam

# Функция для загрузки и предобработки изображения
def load_and_preprocess_image(image_path):
    # Загрузка изображения
    image = Image.open(image_path).convert('RGB')

    # Преобразования для изображения
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Приводим к размеру, который принимает модель
        transforms.ToTensor(),  # Преобразуем в тензор
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Нормализация
    ])

    # Применяем преобразования
    img_tensor = transform(image).unsqueeze(0)  # Добавляем batch dimension
    return img_tensor


# Пути к изображениям
image_paths = [
    "/mnt/data/cfi_manual_glaucoma/train/glaucoma/17255_20240925124036747.png"
]

# Директория для сохранения изображений
output_dir = "/mnt/data/"
os.makedirs(output_dir, exist_ok=True)

# Применение Grad-CAM к каждому изображению
for image_path in image_paths:
    # Загрузка и предобработка изображения
    img_tensor = load_and_preprocess_image(image_path)

    # Определение целевого класса (например, 1 для класса 1)
    target_class = 1  # Меняйте в зависимости от задачи

    # Применение Grad-CAM
    grad_cam_map = grad_cam_vit(model, img_tensor, target_class)

    # Визуализация и сохранение
    plt.figure(figsize=(10, 5))

    # Оригинальное изображение
    plt.subplot(1, 2, 1)
    plt.imshow(Image.open(image_path))
    plt.title("Original Image")
    plt.axis('off')

    # Карта Grad-CAM
    plt.subplot(1, 2, 2)
    plt.imshow(grad_cam_map, cmap='jet')
    plt.title("Grad-CAM")
    plt.axis('off')

    # Сохранение изображений
    image_filename = os.path.basename(image_path)
    grad_cam_filename = f"grad_cam_{image_filename}"

    # Сохраняем визуализации
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, grad_cam_filename))
    plt.close()

print(f"Grad-CAM images saved to: {output_dir}")
