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

model = models_vit.__dict__['vit_large_patch16'](
    num_classes=2,
    drop_path_rate=0.2,
    global_pool=True,
)

checkpoint = torch.load('RETFound_cfp_weights.pth', map_location='cpu')
checkpoint_model = checkpoint['model']
state_dict = model.state_dict()
for k in ['head.weight', 'head.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

interpolate_pos_embed(model, checkpoint_model)

msg = model.load_state_dict(checkpoint_model, strict=False)

assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

trunc_normal_(model.head.weight, std=2e-5)

print("Model = %s" % str(model))


def grad_cam_vit(model, img, target_class):
    model.eval()

    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    hook_forward = model.blocks[-1].register_forward_hook(forward_hook)
    hook_backward = model.blocks[-1].register_full_backward_hook(backward_hook)

    output = model(img)

    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0][target_class] = 1.0
    output.backward(gradient=one_hot)

    assert activations is not None and gradients is not None

    gradients = gradients[:, 1:]
    activations = activations[:, 1:]

    weights = torch.mean(gradients, dim=2)

    grads_cam = torch.einsum('bn,bn->bn', weights, activations.norm(dim=2))  # [batch, num_patches]
    grads_cam = F.relu(grads_cam)

    grads_cam = grads_cam[0].reshape(14, 14).cpu().numpy()  # Для 224x224 и patch_size=16
    grads_cam = cv2.resize(grads_cam, (img.shape[2], img.shape[3]))
    grads_cam = (grads_cam - grads_cam.min()) / (grads_cam.max() - grads_cam.min() + 1e-8)

    hook_forward.remove()
    hook_backward.remove()

    return grads_cam


def load_and_preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


image_path = "/mnt/data/cfi_manual_glaucoma/train/normal/17845_20240614122629638.png"
img_tensor = load_and_preprocess_image(image_path)

with torch.no_grad():
    output = model(img_tensor)
pred_class = output.argmax(dim=1).item()

grad_cam_map = grad_cam_vit(model, img_tensor, pred_class)

plt.figure(figsize=(10, 5))
plt.imshow(Image.open(image_path))
plt.imshow(grad_cam_map, cmap='jet', alpha=0.5)
plt.axis('off')

output_dir = "/mnt/data/"
os.makedirs(output_dir, exist_ok=True)

grad_cam_filename = os.path.join(output_dir, "grad_cam_output.png")
plt.tight_layout()
plt.savefig(grad_cam_filename)
plt.close()

print(f"Grad-CAM image saved to: {grad_cam_filename}")
