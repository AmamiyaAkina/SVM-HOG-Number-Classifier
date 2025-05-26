# train.py

import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

# ------------------ Config ------------------
IMG_SIZE = (28, 28)
TRAIN_DIR = "data/train_images"
TRAIN_LABELS = "data/train_labels.csv"
MODEL_PATH = "model/svm_hog_model.pkl"
TRAIN_SAMPLES = 10000 # Amount of training data

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
    print("📥 Loading training data...")
    X_imgs, y = load_dataset(TRAIN_DIR, TRAIN_LABELS, max_samples=TRAIN_SAMPLES)

    print("🔍 Extracting HOG features...")
    X_hog = extract_hog_features(X_imgs)

    print("⚙️ Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_hog)

    print("🤖 Training SVM model...")
    model = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
    model.fit(X_scaled, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump((model, scaler), MODEL_PATH)
    print(f"✅ Model has been saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
