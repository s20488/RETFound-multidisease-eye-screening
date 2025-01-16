import time
import torch
from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.vision import ModelOnlyCheck
from models_vit import vit_large_patch16

# Глобальные параметры
INPUT_SHAPE = (3, 224, 224)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Инициализация модели
model = vit_large_patch16(global_pool=True).to(DEVICE)


# Функция для создания фиктивного входа
def create_dummy_input(input_shape, device):
    return torch.randn(1, *input_shape).to(device)


# Проверка входных данных
class ModelInputShapeCheck(ModelOnlyCheck):
    def __init__(self, expected_input_shape: tuple, **kwargs):
        super().__init__(**kwargs)
        self.expected_input_shape = expected_input_shape

    def compute(self, context: dict) -> CheckResult:
        model = context['model']
        dummy_input = create_dummy_input(self.expected_input_shape, DEVICE)
        try:
            with torch.no_grad():
                output = model(dummy_input)
            result = {'input_shape_passed': True, 'output_shape': output.shape}
            display = f"Input compatible. Output shape: {output.shape}"
        except Exception as e:
            result = {'input_shape_passed': False, 'error': str(e)}
            display = [f"Input compatible. Output shape: {output.shape}"]
        return CheckResult(result, display=display)

    def add_condition_output_shape(self, expected_output_shape: tuple):
        def condition(result):
            if result.get('input_shape_passed', False):
                category = ConditionCategory.PASS if result[
                                                         'output_shape'] == expected_output_shape else ConditionCategory.FAIL
                message = f"Expected output shape {expected_output_shape}, got {result['output_shape']}."
            else:
                category = ConditionCategory.FAIL
                message = "Input shape check failed; cannot validate output shape."
            return ConditionResult(category, message)

        return self.add_condition("Output shape matches expected", condition)


# Проверка времени инференса
class ModelInferenceTimeCheck(ModelOnlyCheck):
    def compute(self, context: dict) -> CheckResult:
        model = context['model']
        dummy_input = create_dummy_input(INPUT_SHAPE, DEVICE)
        start_time = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        end_time = time.time()
        inference_time = end_time - start_time
        result = {'inference_time': inference_time}
        display = [f"Inference time: {inference_time:.4f} seconds"]
        return CheckResult(result, display=display)

    def add_condition_max_time(self, max_time: float):
        def condition(result):
            category = ConditionCategory.PASS if result['inference_time'] <= max_time else ConditionCategory.FAIL
            message = f"Inference time: {result['inference_time']:.4f} seconds (max {max_time} seconds)."
            return ConditionResult(category, message)

        return self.add_condition("Inference time within limits", condition)


# Проверка распределения весов
class ModelWeightDistributionCheck(ModelOnlyCheck):
    def compute(self, context: dict) -> CheckResult:
        model = context['model']
        weights = [p.data.flatten() for p in model.parameters() if p.requires_grad]
        all_weights = torch.cat(weights)

        mean_weight = all_weights.mean().item()
        std_weight = all_weights.std().item()
        max_weight = all_weights.max().item()
        min_weight = all_weights.min().item()

        result = {
            'mean_weight': mean_weight,
            'std_weight': std_weight,
            'max_weight': max_weight,
            'min_weight': min_weight
        }
        display = [f"Mean: {mean_weight:.4f}, Std: {std_weight:.4f}, Max: {max_weight:.4f}, Min: {min_weight:.4f}"]
        return CheckResult(result, display=display)


# Проверка распределения выходов
class ModelOutputDistributionCheck(ModelOnlyCheck):
    def compute(self, context: dict) -> CheckResult:
        model = context['model']
        dummy_input = create_dummy_input(INPUT_SHAPE, DEVICE)
        with torch.no_grad():
            output = model(dummy_input)

        max_value = output.max().item()
        min_value = output.min().item()
        mean_value = output.mean().item()

        result = {
            'max_value': max_value,
            'min_value': min_value,
            'mean_value': mean_value
        }
        display = [f"Output range: [{min_value:.4f}, {max_value:.4f}]. Mean: {mean_value:.4f}"]
        return CheckResult(result, display=display)


# Контекст передаётся как словарь
context = {'model': model}

# Инициализация проверок
input_shape_check = ModelInputShapeCheck(expected_input_shape=INPUT_SHAPE)
input_shape_check.add_condition_output_shape(expected_output_shape=(1, 1024))

inference_check = ModelInferenceTimeCheck()
inference_check.add_condition_max_time(max_time=0.1)

weight_distribution_check = ModelWeightDistributionCheck()
output_distribution_check = ModelOutputDistributionCheck()

# Запуск проверок
input_shape_result = input_shape_check.run(context)
inference_result = inference_check.run(context)
weight_distribution_result = weight_distribution_check.run(context)
output_distribution_result = output_distribution_check.run(context)

# Вывод результатов
input_shape_result.show()
inference_result.show()
weight_distribution_result.show()
output_distribution_result.show()
