# Images-Classifier

An **image classifier** using the **K-Nearest Neighbors (KNN)** algorithm on the CIFAR-10 dataset.  
This project was created to practice classical machine learning: loading a standard dataset, implementing a nearest-neighbor predictor from scratch, and tuning hyperparameters with scikit-learn.

---

## 🚀 Overview

The project classifies 32×32 color images into 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) using two approaches:

- **Custom Nearest Neighbor** — A from-scratch implementation that compares each test image to all training images using L1 or L2 distance and assigns the label of the closest training example.
- **scikit-learn KNN** — Uses `KNeighborsClassifier` with configurable `k` and distance metric, plus `GridSearchCV` to find optimal hyperparameters (e.g. best `k` and L1 vs L2).

The goal was to reinforce understanding of:

- CIFAR-10 data loading and train/test splitting
- Distance metrics (L1, L2) and nearest-neighbor prediction
- Model evaluation (accuracy) and hyperparameter tuning (cross-validation, grid search)

---

## 🧠 Key Features

- **CIFAR-10 pipeline** — Load batches, split 50k train / 10k test, visualize sample images with labels
- **Nearest Neighbor (1-NN)** — Custom class: train on (X, y), predict by minimizing L1 or L2 distance to training points
- **K-NN with scikit-learn** — `KNeighborsClassifier` with `n_neighbors`, `p` (1=L1, 2=L2), parallelized with `n_jobs=-1`
- **Hyperparameter search** — `GridSearchCV` over `n_neighbors` (e.g. 1, 3, 5, 10, 20, 50, 100) and `p` (1, 2) with 5-fold CV
- **Visualization** — Display single images (OpenCV) or a 5×5 grid of random samples with labels (Matplotlib)

---

## 🛠 Tech Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python" />
</p>

- **Python** — Scripting and logic
- **NumPy** — Array operations and distances
- **OpenCV (cv2)** — Image reshaping and display
- **Matplotlib** — Grid of sample images and labels
- **scikit-learn** — `KNeighborsClassifier`, `GridSearchCV`, `KFold`
- **Pickle** — CIFAR-10 batch loading

---

## 📐 KNN Concepts

**Nearest Neighbor (1-NN):** assign the label of the single closest training example.

![KNN](https://user-images.githubusercontent.com/94462048/212420345-a040b7f1-7ec2-4c48-9b7a-8b7b0f6c7c06.png)

**K-Nearest Neighbors (K-NN):** assign the majority label among the K closest training examples.

![K](https://user-images.githubusercontent.com/94462048/212420722-0bc5b0b5-cc70-4f38-9254-86a3c9fe89d0.png)

---

## 🗂 Project Structure

```
Images_Classifier/
├── README.md
├── cifar10.py                    # CIFAR-10 loader, custom NN, sklearn KNN + GridSearch
└── cifar-10-batches-py/          # CIFAR-10 dataset (data_batch_1..5, test_batch, batches.meta)
    └── readme.html
```

- `cifar10.py` — Defines `CIFAR10` (load, split, `show_img`, `show_examples`), `NearstNeighbor` (custom 1-NN), and runs sklearn KNN + grid search.
- `cifar-10-batches-py/` — Must contain the CIFAR-10 Python version (download from [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) if missing).

---

## 📦 Installation

To run the project locally:

1. Clone the repo:
   ```bash
   git clone https://github.com/initd-fr/Images_Classifier.git
   cd Images_Classifier
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install numpy opencv-python matplotlib scikit-learn
   ```

4. Ensure CIFAR-10 data is present in `cifar-10-batches-py/` (files: `data_batch_1` … `data_batch_5`, `test_batch`, `batches.meta`). If not, download the “CIFAR-10 python version” from [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and extract it into that folder.

5. Run the script:
   ```bash
   python cifar10.py
   ```
   This loads the data, shows a 5×5 sample grid, trains sklearn KNN on the full train set, evaluates on the first 100 test images, and runs grid search (comment/uncomment the custom NN block in the script if you want to compare with 1-NN).

---

## 📌 Notes

- The script trains and evaluates on **100 test samples** by default (`X_test[:100]`); you can change this slice to run on the full test set (slower).
- **GridSearchCV** uses the full training set and 5-fold CV — it can be slow on 50k samples; consider subsampling for quicker experiments.
- CIFAR-10 is not included in the repo by default; you must download and place the batch files in `cifar-10-batches-py/` as described above.
