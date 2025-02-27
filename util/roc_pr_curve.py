import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score


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

    unique_labels = np.unique(true_labels)

    if len(unique_labels) < 2:
        print(f"Error: Only one class ({unique_labels}) in the data. Unable to calculate the AUC-PR.")
        return

    plt.figure()

    if num_class > 2:
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
        auc_roc = roc_auc_score(true_labels, predicted_probs[:, 0])

        fpr, tpr, _ = roc_curve(true_labels, predicted_probs[:, 0])
        plt.plot(fpr, tpr, label=f'{class_names[0]} (AUC = {auc_roc:.2f})')

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

    unique_labels = np.unique(true_labels)

    if len(unique_labels) < 2:
        print(f"Error: Only one class ({unique_labels}) in the data. Unable to calculate the AUC-PR.")
        return

    plt.figure()

    if num_class > 2:
        for i in range(num_class):
            auc_pr_class = average_precision_score(true_labels_onehot[:, i], predicted_probs[:, i])

            precision, recall, _ = precision_recall_curve(true_labels_onehot[:, i], predicted_probs[:, i])
            plt.plot(recall, precision, label=f'{class_names[i]} (AUC = {auc_pr_class:.4f})')

    else:
        positive_class_index = 0
        positive_probs = predicted_probs[:, positive_class_index] if predicted_probs.ndim == 2 else predicted_probs
        auc_pr_class = average_precision_score(true_labels == positive_class_index, positive_probs)

        precision, recall, _ = precision_recall_curve(true_labels == positive_class_index, positive_probs)
        plt.plot(recall, precision, label=f'{class_names[positive_class_index]} (AUC = {auc_pr_class:.4f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left', fontsize='small')
    plt.grid()

    os.makedirs(task, exist_ok=True)
    save_path = os.path.join(task, "precision_recall_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
