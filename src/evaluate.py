import os
import numpy as np
import torch
import torch.nn.functional as F
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm

from src.config import *
from src.utils import get_keras_loader, get_pytorch_loader
from src.metrics import print_metrics, plot_roc_curve
from src.train import CNNModel, build_keras_model


def evaluate_keras():
    """
    Load and evaluate the Keras model.
    """

    # FIX: use WEIGHTS path (not full .keras model)
    if not os.path.exists(KERAS_WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Keras weights not found at {KERAS_WEIGHTS_PATH}"
        )

    print("\nLoading Keras model...")

    loader = get_keras_loader()

    # Rebuild architecture first (correct approach)
    model = build_keras_model()

    # Load only weights (fixes deserialization issues)
    model.load_weights(KERAS_WEIGHTS_PATH)

    print("Running Keras inference...")

    probabilities = model.predict(loader, verbose=0)

    # Safe binary classification handling
    probabilities = probabilities[:, 1]

    predictions = (probabilities > 0.5).astype(int)

    labels = loader.classes

    return labels, predictions, probabilities


def evaluate_pytorch():
    """
    Load and evaluate the PyTorch model.
    """

    if not os.path.exists(PYTORCH_MODEL_PATH):
        raise FileNotFoundError(
            f"PyTorch model not found at {PYTORCH_MODEL_PATH}"
        )

    print("\nLoading PyTorch model...")

    loader = get_pytorch_loader()

    model = CNNModel().to(DEVICE)

    model.load_state_dict(
        torch.load(PYTORCH_MODEL_PATH, map_location=DEVICE)
    )

    model.eval()

    all_predictions = []
    all_probabilities = []
    all_labels = []

    print("Running PyTorch inference...")

    with torch.no_grad():
        for images, labels in tqdm(loader):

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = F.softmax(outputs, dim=1)[:, 1]
            predictions = torch.argmax(outputs, dim=1)

            all_probabilities.extend(probabilities.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )


def compare_models():
    """
    Evaluate both models and compare results.
    """

    keras_labels, keras_preds, keras_probs = evaluate_keras()
    pytorch_labels, pytorch_preds, pytorch_probs = evaluate_pytorch()

    print_metrics(
        keras_labels,
        keras_preds,
        keras_probs,
        CLASS_LABELS,
        "Keras Model"
    )

    print_metrics(
        pytorch_labels,
        pytorch_preds,
        pytorch_probs,
        CLASS_LABELS,
        "PyTorch Model"
    )

    plt.figure(figsize=(8, 6))

    plot_roc_curve(keras_labels, keras_probs, "Keras")
    plot_roc_curve(pytorch_labels, pytorch_probs, "PyTorch")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    compare_models()