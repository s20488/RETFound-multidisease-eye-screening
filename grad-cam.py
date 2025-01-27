import torch
import numpy as np
import os
from torchvision import transforms
from PIL import Image
from models_vit import vit_large_patch16


# Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Регистрация хуков
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_module = dict(self.model.named_modules())[self.target_layer]
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, class_index=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        class_index = class_index if class_index is not None else torch.argmax(output, dim=1).item()
        loss = output[:, class_index]
        loss.backward()

        pooled_grads = torch.mean(self.gradients, dim=(0, 2, 3))
        activations = self.activations.squeeze(0)
        heatmap = torch.einsum("chw,c->hw", activations, pooled_grads).detach().cpu().numpy()
        return np.maximum(heatmap, 0) / np.max(heatmap)  # ReLU + нормализация


# Предобработка изображения
def preprocess_image(image_path, target_size=(224, 224)):
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    return preprocess(img).unsqueeze(0), img


# Сохранение тепловой карты
def save_heatmap(image, heatmap, save_path, alpha=0.6):
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(image.size, resample=Image.BICUBIC)
    heatmap = np.expand_dims(np.array(heatmap), axis=2)
    heatmap = np.concatenate([heatmap] * 3, axis=2)
    overlay = np.uint8(alpha * heatmap + (1 - alpha) * np.array(image))
    Image.fromarray(overlay).save(save_path)
    print(f"Сохранено: {save_path}")


# Основной код
if __name__ == "__main__":
    # Загрузка модели
    model = vit_large_patch16(global_pool=True)
    model.load_state_dict(torch.load(
        '/mnt/data/Anastasiia_Ponkratova/RETFound_MAE/results/finetune_cfi_manual_hypertension_strict/checkpoint-best.pth',
        map_location=torch.device('gpu')
    ))
    model.eval()

    # Пути к изображениям
    image_paths = [
        "/mnt/data/cfi_manual_hypertension_0.1/train/1/17947_20230801130127534.png",
        "/mnt/data/cfi_manual_hypertension_0.1/train/0/11929_20210225080615244.png",
        "/mnt/data/cfi_manual_hypertension_0.1/val/0/12134_20220112141845970.png",
        "/mnt/data/cfi_manual_hypertension_0.1/val/1/10098_20210225071827175.png",
        "/mnt/data/cfi_manual_hypertension_0.1/test/1/12131_20210929141207205.png"
    ]

    # Папка для сохранения результатов
    output_folder = "heatmap_results/"
    os.makedirs(output_folder, exist_ok=True)

    # Grad-CAM
    grad_cam = GradCAM(model, target_layer='blocks.23')  # Убедись, что слой существует

    for image_path in image_paths:
        try:
            # Подготовка данных
            img_tensor, original_image = preprocess_image(image_path)

            # Генерация тепловой карты
            heatmap = grad_cam.generate_heatmap(img_tensor)

            # Сохранение изображения с тепловой картой
            save_path = os.path.join(output_folder, os.path.basename(image_path).replace(".png", "_heatmap.png"))
            save_heatmap(original_image, heatmap, save_path)
        except Exception as e:
            print(f"Ошибка обработки {image_path}: {e}")
