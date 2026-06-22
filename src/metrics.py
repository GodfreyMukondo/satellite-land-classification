import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve
)


def calculate_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob)
    }


def print_metrics(y_true, y_pred, y_prob, labels, model_name):
    metrics = calculate_metrics(y_true, y_pred, y_prob)

    print(f"\n{model_name} Evaluation Results")
    print("=" * 50)

    for key, value in metrics.items():
        print(f"{key.upper()}: {value:.4f}")

    print("\nClassification Report")
    print(classification_report(y_true, y_pred, target_names=labels))

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    disp.plot()
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()


def plot_roc_curve(y_true, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} AUC={auc:.4f}"
    )