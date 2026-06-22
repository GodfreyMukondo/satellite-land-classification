import os
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf

from src.config import *
from src.utils import get_pytorch_loader, get_keras_loader


# ============================================================
# PYTORCH MODEL
# ============================================================

class CNNModel(nn.Module):
    """
    PyTorch CNN for satellite classification (agri vs non-agri)
    """

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# KERAS MODEL
# ============================================================

def build_keras_model():
    """
    Build Keras CNN model (architecture only)
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            32, (5, 5),
            padding="same",
            activation="relu",
            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Conv2D(64, (5, 5), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Conv2D(128, (5, 5), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(2, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# TRAIN KERAS
# ============================================================

def train_keras():
    print("\nStarting TensorFlow/Keras training...\n")

    train_loader = get_keras_loader()
    model = build_keras_model()

    model.fit(
        train_loader,
        epochs=EPOCHS,
        verbose=1
    )

    # SAVE ONLY WEIGHTS (FIXED)
    model.save_weights(KERAS_WEIGHTS_PATH)

    print("\nKeras training completed.")
    print(f"Saved weights to: {KERAS_WEIGHTS_PATH}")


# ============================================================
# TRAIN PYTORCH
# ============================================================

def train_pytorch():
    print(f"\nStarting PyTorch training on {DEVICE}...\n")

    loader = get_pytorch_loader()
    model = CNNModel().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Loss: {running_loss/len(loader):.4f}"
        )

    torch.save(model.state_dict(), PYTORCH_MODEL_PATH)

    print("\nPyTorch training completed.")
    print(f"Saved to: {PYTORCH_MODEL_PATH}")


# ============================================================
# MASTER TRAIN
# ============================================================

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_keras()
    train_pytorch()

    print("\nAll models trained successfully.")


if __name__ == "__main__":
    train()