# UE24CS645BC2_PES1PG25CS105_Fashion_MNIST_CNN

## Overview

This project implements a **Convolutional Neural Network (CNN) from scratch using NumPy** without using deep learning frameworks like TensorFlow or PyTorch.

The CNN is trained on the **Fashion MNIST dataset**, which contains grayscale images of clothing items belonging to 10 different classes.

The project demonstrates:
- Forward propagation
- Backpropagation
- Convolution operations
- Max Pooling
- Fully Connected layers
- Softmax classification
- Cross Entropy Loss

---

# Dataset

Fashion MNIST Dataset:
- 70,000 grayscale images
- Image size: 28 × 28
- 10 classes of clothing items

Classes:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

---

# Project Structure

```text
UE24CS645BC2_PES1PG25CS105_Fashion_MNIST_CNN/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── model.py
│   ├── train.py
│   │
│   ├── layers/
│   │   ├── conv_layer.py
│   │   ├── pool_layer.py
│   │   ├── flatten_layer.py
│   │   ├── fc_layer.py
│   │   └── __init__.py
│   │
│   ├── utils/
│   │   ├── data_loader.py
│   │   └── __init__.py
│
└── results.png
```

---

# CNN Architecture

```text
Input Image (1 × 28 × 28)
        │
        ▼
Conv Layer (1 → 8 filters, 3×3)
        │
        ▼
ReLU Activation
        │
        ▼
Max Pooling (2×2)
        │
        ▼
Conv Layer (8 → 16 filters, 3×3)
        │
        ▼
ReLU Activation
        │
        ▼
Max Pooling (2×2)
        │
        ▼
Flatten Layer
        │
        ▼
Fully Connected Layer (784 → 128)
        │
        ▼
ReLU Activation
        │
        ▼
Fully Connected Layer (128 → 10)
        │
        ▼
Softmax Output
```

---

# Layers Implemented

| Layer | Description |
|---|---|
| Convolution Layer | Feature extraction using filters |
| ReLU Activation | Non-linear activation |
| Max Pooling | Downsampling operation |
| Flatten Layer | Converts 3D tensor to 1D vector |
| Fully Connected Layer | Dense neural network layer |
| Softmax Layer | Multi-class classification |

---

# Technologies Used

- Python
- NumPy
- Matplotlib
- Scikit-learn
- tqdm

---

# Installation

## Step 1 — Clone Repository

```bash
git clone https://github.com/<your-username>/UE24CS645BC2_PES1PG25CS105_Fashion_MNIST_CNN.git
```

---

## Step 2 — Navigate to Project Folder

```bash
cd UE24CS645BC2_PES1PG25CS105_Fashion_MNIST_CNN
```

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

Move into the `src` folder:

```bash
cd src
```

Run the training script:

```bash
python train.py
```

---

# Output

During execution:
- Dataset will be loaded
- CNN will train for multiple epochs
- Loss and accuracy will be displayed

After training:
- Accuracy graph and loss graph will be saved as:

```text
results.png
```

---

# Sample Results

| Metric | Value |
|---|---|
| Training Accuracy | ~75% |
| Testing Accuracy | ~72–78% |

(Note: Accuracy may vary depending on initialization and training conditions.)

---

# Learning Outcomes

This project helped in understanding:
- Internal working of CNNs
- Manual implementation of backpropagation
- Gradient computation
- Feature extraction using convolution
- Neural network training without deep learning frameworks

---

# Author

## Name
Karunya M Adekar

## USN
PES1PG25CS105

---

# Submission Details

Subject:
```text
DLTP_Assignment_1
```

Submission includes:
- GitHub Repository Link
- Source Code
- README File
- Training Results

---

# References

- Fashion MNIST Dataset
- NumPy Documentation
- Scikit-learn Documentation
- Deep Learning Fundamentals