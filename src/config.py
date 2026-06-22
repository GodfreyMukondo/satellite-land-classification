import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "images_dataSAT")
MODEL_DIR = os.path.join(BASE_DIR, "models")

KERAS_WEIGHTS_PATH = os.path.join(MODEL_DIR, "keras_model.weights.h5")
PYTORCH_MODEL_PATH = os.path.join(MODEL_DIR, "pytorch_model.pth")

IMG_WIDTH = 64
IMG_HEIGHT = 64
BATCH_SIZE = 128

CLASS_LABELS = ["non-agri", "agri"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEARNING_RATE = 0.001
EPOCHS = 10