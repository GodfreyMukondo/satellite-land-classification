# 🌍 Satellite Land Classification (Agri vs Non-Agri)

This project uses **Deep Learning (CNN models in TensorFlow/Keras and PyTorch)** to classify satellite images into:

- 🌾 Agricultural land (agri)
- 🏙️ Non-agricultural land (non-agri)

The system trains and compares both frameworks to evaluate performance.

---

## 📊 Dataset

- Total images: **6000**
- Classes:
  - agri
  - non-agri

Images are loaded using custom PyTorch and Keras data loaders.

---

## 🧠 Models Used

### 1. TensorFlow / Keras CNN

**Architecture:**

- Conv2D → MaxPooling
- Conv2D → MaxPooling
- Conv2D → MaxPooling
- GlobalAveragePooling
- Dense (64)
- Dropout (0.3)
- Dense (Softmax output)

**Training Settings:**

- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Epochs: 10

**Training Results:**

- Final accuracy: ~98–99%
- Final loss: ~0.03–0.04

---

### 2. PyTorch CNN

**Architecture:**

- 3 Convolutional blocks
- ReLU activation
- MaxPooling layers
- Adaptive Average Pooling
- Fully connected layers
- Dropout (0.3)

**Training Settings:**

- Optimizer: Adam
- Loss: CrossEntropyLoss
- Epochs: 10

**Training Results:**

- Final loss: ~0.015–0.03
- Stable convergence

---

## ⚙️ Training Logs Summary

During training:

- TensorFlow trained on CPU (Windows limitation)
- Keras achieved high accuracy (~98%+)
- PyTorch showed consistent loss reduction
- Both models trained successfully and saved

---

## 💾 Saved Models

models/
├── keras_model.weights.h5
└── pytorch_model.pth

---

## 🚀 How to Run

bash
git clone https://github.com/GodfreyMukondo/satellite-land-classification.git
cd satellite-land-classification

#Create virtual environment:
python -m venv .venv
.\.venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Train models:
python -m src.train

Evaluate models:
python -m src.evaluate

📈 Evaluation

The system provides:

Accuracy comparison (Keras vs PyTorch)
Precision, recall, F1-score
ROC curve comparison
Probability-based predictions
⚠️ Notes
TensorFlow GPU is not enabled on native Windows (CPU used instead)
Dataset and models are excluded from GitHub via .gitignore
Training uses 64×64 satellite image inputs
📁 Project Structure
satellite-land-classification/
│
├── data/ # Dataset (ignored in GitHub)
├── models/ # Saved models (ignored)
├── notebooks/ # Experiments & analysis
├── src/ # Source code
│ ├── train.py
│ ├── evaluate.py
│ ├── model.py
│ ├── utils.py
│ ├── config.py
│
├── requirements.txt
├── .gitignore
└── README.md

👨‍💻 Author: Godfrey Mukondo

⭐ Outcome

This project demonstrates:

End-to-end ML pipeline
Dual-framework implementation (PyTorch + TensorFlow)
Model comparison techniques
Real satellite image classification
