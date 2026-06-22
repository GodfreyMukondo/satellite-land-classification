import os
import torch
import tensorflow as tf

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from src.config import *


def create_directories():
    """
    Create required project directories.
    """

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def get_keras_loader():
    """
    Create TensorFlow/Keras data loader.
    """

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255
    )

    loader = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True
    )

    return loader


def get_pytorch_loader():
    """
    Create PyTorch data loader.
    """

    transform = transforms.Compose([
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    dataset = datasets.ImageFolder(
        root=DATA_DIR,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    return loader