import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


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
        softmax_probs = torch.nn.Softmax(dim=1)(outputs)

        true_labels.extend(targets.cpu().numpy())
        predicted_probs.extend(softmax_probs.cpu().numpy())

    true_labels = np.array(true_labels)
    predicted_probs = np.array(predicted_probs)

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

    plt.figure()
    for i in range(num_class):
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')  # Use class names

    plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})', linestyle='--')
    plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})', linestyle=':')
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
        class_names = dataset.classes
    else:
        class_names = [f'Class {i}' for i in range(num_class)]

    for batch in data_loader:
        images = batch[0].to(device, non_blocking=True)
        targets = batch[-1].to(device, non_blocking=True)

        outputs = model(images)
        softmax_probs = torch.nn.Softmax(dim=1)(outputs)

        true_labels.extend(targets.cpu().numpy())
        predicted_probs.extend(softmax_probs.cpu().numpy())

    true_labels = np.array(true_labels)
    predicted_probs = np.array(predicted_probs)

    true_labels_onehot = np.eye(num_class)[true_labels]

    precision = {}
    recall = {}
    pr_auc = {}

    for i in range(num_class):
        precision[i], recall[i], _ = precision_recall_curve(true_labels_onehot[:, i], predicted_probs[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    precision["micro"], recall["micro"], _ = precision_recall_curve(true_labels_onehot.ravel(), predicted_probs.ravel())
    pr_auc["micro"] = auc(recall["micro"], precision["micro"])

    all_recall = np.unique(np.concatenate([recall[i] for i in range(num_class)]))
    mean_precision = np.zeros_like(all_recall)

    for i in range(num_class):
        mean_precision += np.interp(all_recall, recall[i][::-1], precision[i][::-1])

    mean_precision /= num_class
    recall["macro"] = all_recall
    precision["macro"] = mean_precision
    pr_auc["macro"] = auc(recall["macro"], precision["macro"])

    plt.figure()
    for i in range(num_class):
        plt.plot(recall[i], precision[i], label=f'{class_names[i]} (AUC = {pr_auc[i]:.2f})')

    plt.plot(recall["micro"], precision["micro"], label=f'Micro-average (AUC = {pr_auc["micro"]:.2f})', linestyle='--')
    plt.plot(recall["macro"], precision["macro"], label=f'Macro-average (AUC = {pr_auc["macro"]:.2f})', linestyle=':')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left', fontsize='small')
    plt.grid()

    os.makedirs(task, exist_ok=True)
    save_path = os.path.join(task, "precision_recall_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
