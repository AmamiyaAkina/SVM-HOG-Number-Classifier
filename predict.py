# predict.py

import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ------------------ Config ------------------
IMG_SIZE = (28, 28)
TEST_DIR = "data/test_images"
TEST_LABELS = "data/test_labels.csv"
MODEL_PATH = "model/svm_hog_model.pkl"
TEST_SAMPLES = 2000 # Amount of testing data

# ------------------ Data Loading ------------------
def load_dataset(image_dir, label_csv, max_samples=None):
    df = pd.read_csv(label_csv)
    df.columns = df.columns.str.strip().str.lower()
    if max_samples:
        df = df[:max_samples]

    images, labels = [], []
    for _, row in df.iterrows():
        fname = row['filename']
        label = int(row['label'])
        path = os.path.join(image_dir, fname)
        if not os.path.exists(path):
            continue
        img = Image.open(path).convert('L').resize(IMG_SIZE)
        images.append(np.array(img))
        labels.append(label)

    return np.array(images), np.array(labels)

# ------------------ HOG features ------------------
def extract_hog_features(images):
    return np.array([
        hog(img, pixels_per_cell=(4, 4), cells_per_block=(2, 2),
            orientations=9, block_norm='L2-Hys')
        for img in images
    ])

# ------------------ Main Function ------------------
def main():
    if not os.path.exists(MODEL_PATH):
        print("❌ Model file not found, please run train.py first for initial model training")
        return

    print("📦 Loading model...")
    model, scaler = joblib.load(MODEL_PATH)

    print("📥 Loading test data...")
    X_imgs, y_true = load_dataset(TEST_DIR, TEST_LABELS, max_samples=TEST_SAMPLES)

    print("🔍 Extracting HOG features...")
    X_hog = extract_hog_features(X_imgs)
    X_scaled = scaler.transform(X_hog)

    print("🔮 Predicting...")
    y_pred = model.predict(X_scaled)

    print("🎯 Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    main()
