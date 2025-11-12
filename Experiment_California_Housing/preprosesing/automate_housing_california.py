import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import joblib 


BASE_DIR = "california_housing_data"
RAW_DATA_PATH = os.path.join(BASE_DIR, "california_housing.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "namadataset_preprocessing")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.joblib")
TRAIN_PATH = os.path.join(OUTPUT_DIR, "train.csv")
TEST_PATH = os.path.join(OUTPUT_DIR, "test.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_raw_data():
    """
    Fungsi untuk memuat data mentah.
    GANTI FUNGSI INI untuk memuat dataset Anda (misal: pd.read_csv(...)).
    Di sini kita simulasi dengan membuat file data mentah.
    """
    print("Memuat data mentah...")
    if not os.path.exists(RAW_DATA_PATH):
        print("Data mentah tidak ditemukan, membuat data mentah simulasi...")
        housing = fetch_california_housing()
        data = pd.DataFrame(housing.data, columns=housing.feature_names)
        data['MedHouseVal'] = housing.target
        data.to_csv(RAW_DATA_PATH, index=False)
        print(f"Data mentah simulasi disimpan di {RAW_DATA_PATH}")
    
    return pd.read_csv(RAW_DATA_PATH)

def preprocess_data(data):
    """
    Melakukan preprocessing data: pemisahan, scaling, dan penyimpanan.
    """
    print("Memulai preprocessing...")
    
    # 1. Pisahkan Fitur dan Target
    X = data.drop('MedHouseVal', axis=1)
    y = data['MedHouseVal']
    
    # 2. Pembagian Data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Scaling
    scaler = StandardScaler()
    
    # Fit HANYA pada data train
    scaler.fit(X_train_raw)
    
    # Transform
    X_train_scaled = scaler.transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # 4. Gabungkan kembali data yang sudah diproses
    train_data = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_data['MedHouseVal'] = y_train.values

    test_data = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_data['MedHouseVal'] = y_test.values

    # 5. Buat folder output jika belum ada
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 6. Simpan data yang diproses dan scaler
    train_data.to_csv(TRAIN_PATH, index=False)
    test_data.to_csv(TEST_PATH, index=False)
    joblib.dump(scaler, SCALER_PATH)
    
    print(f"Preprocessing selesai.")
    print(f"Data train disimpan di: {TRAIN_PATH}")
    print(f"Data test disimpan di: {TEST_PATH}")
    print(f"Scaler disimpan di: {SCALER_PATH}")
    
    return train_data, test_data

def main():
    """
    Fungsi utama untuk menjalankan pipeline.
    """
    raw_data = load_raw_data()
    preprocess_data(raw_data)

if __name__ == "__main__":
    main()
