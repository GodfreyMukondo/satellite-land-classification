import torch
import torch.nn as nn
import tensorflow as tf

from src.config import IMG_WIDTH, IMG_HEIGHT, LEARNING_RATE


# ============================================================
# PYTORCH MODEL
# ============================================================

class CNNModel(nn.Module):
    """
    PyTorch CNN model for satellite land classification.
    Classes: agri vs non-agri
    """

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),

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
    Build CNN model for TensorFlow/Keras
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