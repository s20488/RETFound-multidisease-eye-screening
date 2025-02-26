import numpy as np

# Загружаем данные
eval_true_labels = np.load("path/to/task/eval_true_labels.npy")
eval_predicted_probs = np.load("path/to/task/eval_predicted_probs.npy")
plot_true_labels = np.load("path/to/task/plot_true_labels.npy")
plot_predicted_probs = np.load("path/to/task/plot_predicted_probs.npy")

# Сравниваем true_labels
print("True labels match:", np.array_equal(eval_true_labels, plot_true_labels))

# Сравниваем predicted_probs
print("Predicted probabilities match:", np.allclose(eval_predicted_probs, plot_predicted_probs))
