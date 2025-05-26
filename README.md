# SVM-HOG-Number-Classifier

🎯 A lightweight handwritten digit recognition system based on **HOG feature extraction** and **SVM classification**, supporting training, testing, and model reuse.

---

## 📦 Project Overview

This project implements a handwritten digit classifier using:

- 🧠 **HOG (Histogram of Oriented Gradients)** for feature extraction  
- ⚙️ **SVM (Support Vector Machine)** with class balancing  
- 🗂 Structured datasets (images + CSV labels)

---

## 🛠 Installation

Requires Python 3.8+  
Install dependencies with:

```bash
pip install -r requirements.txt
```
Sample requirements.txt:
```
numpy
pandas
scikit-learn
scikit-image
Pillow
joblib
```

🚀 Usage

✅ Deploy data source

Extract the `Data.zip` into the project.

✅ Train the Model

python train.py

    Loads images from data/train_images/
    Reads labels from train_labels.csv
    Extracts HOG features
    Trains SVM classifier
    Saves model to model/svm_hog_model.pkl

✅ Evaluate the Model

python predict.py

    Loads saved model
    Processes test_images/ with HOG
    Outputs accuracy, precision, recall
