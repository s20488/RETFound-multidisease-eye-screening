import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


@torch.no_grad()
def plot_roc_curve(data_loader, model, device, num_class, task):
    model.eval()
    true_labels = []
    predicted_probs = []

    dataset = data_loader.dataset
    if hasattr(dataset, 'classes'):
        class_names = dataset.classes
    else:
        class_names = [f'Class {i}' for i in range(num_class)]

    for batch in data_loader:
        images = batch[0].to(device, non_blocking=True)
        targets = batch[-1].to(device, non_blocking=True)

        outputs = model(images)
        softmax_probs = torch.nn.Softmax(dim=1)(outputs) if num_class > 2 else torch.sigmoid(outputs)

        true_labels.extend(targets.cpu().numpy())
        predicted_probs.extend(softmax_probs.cpu().numpy())

    true_labels = np.array(true_labels)
    predicted_probs = np.array(predicted_probs)

    plt.figure()

    if num_class > 2:
        true_labels_onehot = np.eye(num_class)[true_labels]

        fpr = {}
        tpr = {}
        roc_auc = {}

        for i in range(num_class):
            fpr[i], tpr[i], _ = roc_curve(true_labels_onehot[:, i], predicted_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(true_labels_onehot.ravel(), predicted_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(num_class):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= num_class
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        for i in range(num_class):
            plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

        plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})', linestyle='--')
        plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})', linestyle=':')
    else:
        positive_probs = predicted_probs[:, 1]
        fpr, tpr, _ = roc_curve(true_labels, positive_probs)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'{class_names[0]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random classifier')
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right', fontsize='small')
    plt.grid()

    os.makedirs(task, exist_ok=True)
    save_path = os.path.join(task, "roc_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def plot_pr_curve(data_loader, model, device, num_class, task):
    model.eval()
    true_labels = []
    predicted_probs = []

    dataset = data_loader.dataset
    if hasattr(dataset, 'classes'):
        class_names = dataset.classes  # ["hypertension", "normal"]
        print(f"Class names: {class_names}")  # Проверка порядка классов
    else:
        class_names = [f'Class {i}' for i in range(num_class)]

    # Собираем предсказания и истинные метки
    for batch in data_loader:
        images = batch[0].to(device, non_blocking=True)
        targets = batch[-1].to(device, non_blocking=True)

        outputs = model(images)
        # Используем Softmax для многоклассовой классификации и Sigmoid для бинарной
        if num_class > 2:
            softmax_probs = torch.nn.Softmax(dim=1)(outputs)
        else:
            softmax_probs = torch.sigmoid(outputs)

        true_labels.extend(targets.cpu().numpy())
        predicted_probs.extend(softmax_probs.cpu().numpy())

    true_labels = np.array(true_labels)
    predicted_probs = np.array(predicted_probs)

    # Преобразуем метки в one-hot encoding, как в evaluate
    true_labels_onehot = np.eye(num_class)[true_labels]

    # Проверка наличия объектов обоих классов
    unique_labels = np.unique(true_labels)
    print(f"Unique labels: {unique_labels}")  # Проверка уникальных меток
    if len(unique_labels) < 2:
        print(f"Ошибка: В данных только один класс ({unique_labels}). Невозможно рассчитать AUC-PR.")
        return

    plt.figure()

    if num_class > 2:
        # Для многоклассовой классификации используем average_precision_score с average='macro'
        auc_pr_macro = average_precision_score(true_labels_onehot, predicted_probs, average='macro')
        print(f"Macro-average AUC-PR: {auc_pr_macro:.4f}")

        # Рассчитываем AUC-PR для каждого класса отдельно
        for i in range(num_class):
            auc_pr_class = average_precision_score(true_labels_onehot[:, i], predicted_probs[:, i])
            print(f"Class {class_names[i]} - AUC-PR: {auc_pr_class:.4f}")

            # Строим кривую Precision-Recall для каждого класса
            precision, recall, _ = precision_recall_curve(true_labels_onehot[:, i], predicted_probs[:, i])
            plt.plot(recall, precision, label=f'{class_names[i]} (AUC = {auc_pr_class:.4f})')

    else:
        # Для бинарной классификации используем average_precision_score
        # Определяем индекс класса "hypertension" (первый класс в списке)
        positive_class_index = 0  # "hypertension" соответствует индексу 0
        print(f"Positive class index: {positive_class_index}")  # Проверка индекса положительного класса

        # Выбираем вероятности для положительного класса
        positive_probs = predicted_probs[:, positive_class_index] if predicted_probs.ndim == 2 else predicted_probs
        print(f"Positive probabilities shape: {positive_probs.shape}, content: {positive_probs[:10]}")  # Первые 10 вероятностей

        # Рассчитываем AUC-PR
        auc_pr = average_precision_score(true_labels, positive_probs)
        print(f"Class {class_names[positive_class_index]} - AUC-PR: {auc_pr:.4f}")

        # Строим кривую Precision-Recall
        precision, recall, _ = precision_recall_curve(true_labels, positive_probs)
        plt.plot(recall, precision, label=f'{class_names[positive_class_index]} (AUC = {auc_pr:.4f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left', fontsize='small')
    plt.grid()

    # Сохраняем график
    os.makedirs(task, exist_ok=True)
    save_path = os.path.join(task, "precision_recall_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
