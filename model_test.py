# model_test.py
import torch
from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.vision import ModelOnlyCheck
from models_vit import vit_large_patch16  # Импорт модели из вашего проекта
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm.data import create_transform
import os

# Глобальные параметры
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
INPUT_SHAPE = (3, 224, 224)

# Средние значения и стандартные отклонения для ImageNet
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]  # Среднее значение
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]   # Стандартное отклонение

# Функция для создания фиктивного входа
def create_dummy_input(input_shape, device):
    return torch.randn(1, *input_shape).to(device)

# Пример функции для загрузки датасетов
def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    if is_train == 'train':
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC))
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

class ModelInputShapeCheck(ModelOnlyCheck):
    def __init__(self, expected_input_shape: tuple, **kwargs):
        super().__init__(**kwargs)
        self.expected_input_shape = expected_input_shape

    def compute(self, context: dict) -> CheckResult:
        model = context['model']
        dataset = context['dataset']
        dummy_input = create_dummy_input(self.expected_input_shape, DEVICE)

        try:
            with torch.no_grad():
                # Используем DataLoader для инференса
                for data in dataset:
                    input_data, _ = data
                    output = model(input_data.to(DEVICE))  # data[0] — это изображение, предполагая, что dataset возвращает пару (input, target)
                    break  # Останавливаемся на первом батче для проверки
            result = {'input_shape_passed': True, 'output_shape': output.shape}
            display = f"Input compatible. Output shape: {output.shape}"
        except Exception as e:
            result = {'input_shape_passed': False, 'error': str(e)}
            display = [f"Input compatible. Output shape: {output.shape}"]
        return CheckResult(result, display=display)

    def add_condition_output_shape(self, expected_output_shape: tuple):
        def condition(result):
            if result.get('input_shape_passed', False):
                category = ConditionCategory.PASS if result['output_shape'] == expected_output_shape else ConditionCategory.FAIL
                message = f"Expected output shape {expected_output_shape}, got {result['output_shape']}."
            else:
                category = ConditionCategory.FAIL
                message = "Input shape check failed; cannot validate output shape."
            return ConditionResult(category, message)

        return self.add_condition("Output shape matches expected", condition)


if __name__ == "__main__":
    # Параметры и модель
    class Args:
        data_path = r"C:\Users\Anastasiia\Desktop\Praca_dyplomowa\Praktyka\RETFound_MAE\ORIGA"
        input_size = 224
        color_jitter = 0.4
        aa = 'rand-m9-mstd0.5'
        reprob = 0.25
        remode = 'pixel'
        recount = 1
        batch_size = 16
        num_workers = 4
        pin_mem = True

    args = Args()

    # Загрузка датасетов
    dataset_train = build_dataset(is_train='train', args=args)
    dataset_val = build_dataset(is_train='val', args=args)
    dataset_test = build_dataset(is_train='test', args=args)

    # Создайте DataLoader
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = vit_large_patch16(global_pool=True).to(DEVICE)

    # Передача модели и датасета в контекст
    context = {'model': model, 'dataset': dataloader_test}

    # Инициализация и запуск проверки
    input_shape_check = ModelInputShapeCheck(expected_input_shape=INPUT_SHAPE)
    input_shape_check.add_condition_output_shape(expected_output_shape=(1, 1024))

    result = input_shape_check.run(context)
    result.show()
