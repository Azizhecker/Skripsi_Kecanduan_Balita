import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(
    BASE_DIR,
    "tabulasi data2.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model_random_forest.pkl"
)


# ============================================================
# 1. BACA DATASET
# ============================================================

print("=" * 60)
print("MEMBACA DATASET")
print("=" * 60)

df = pd.read_csv(CSV_PATH)

print("Jumlah data :", len(df))
print("Jumlah kolom:", len(df.columns))


# ============================================================
# 2. RENAME KOLOM
# ============================================================

df = df.rename(columns={
    "Unnamed: 0": "No_Responden",
    "Unnamed: 21": "SKOR_TOTAL_Y",
    "Unnamed: 22": "LABEL"
})


# ============================================================
# 3. HAPUS DATA DENGAN LABEL KOSONG
# ============================================================

df = df.dropna(
    subset=["LABEL"]
).reset_index(drop=True)

print("Data setelah pembersihan:", len(df))


# ============================================================
# 4. DEFINISI ITEM X1-X4
# ============================================================

x1_cols = [
    "X1.1",
    "X1.2",
    "X1.3"
]

x2_cols = [
    "X2.1",
    "X2.2",
    "X2.3"
]

x3_cols = [
    "X3.1",
    "X3.2",
    "X3.3"
]

x4_cols = [
    "X4.1",
    "X4.2",
    "X4.3"
]

y_cols = [
    "Y1",
    "Y2",
    "Y3",
    "Y4"
]


# ============================================================
# 5. HITUNG SKOR X1-X4
# ============================================================

df["Skor_X1"] = df[x1_cols].sum(axis=1)

df["Skor_X2"] = df[x2_cols].sum(axis=1)

df["Skor_X3"] = df[x3_cols].sum(axis=1)

df["Skor_X4"] = df[x4_cols].sum(axis=1)


# ============================================================
# 6. HITUNG TOTAL Y
# ============================================================

df["Total_Y"] = df[y_cols].sum(axis=1)


# ============================================================
# 7. FITUR MODEL
# ============================================================

feature_cols = [
    "Skor_X1",
    "Skor_X2",
    "Skor_X3",
    "Skor_X4"
]

X = df[feature_cols]

y = df["LABEL"]


print("\n" + "=" * 60)
print("FITUR RANDOM FOREST")
print("=" * 60)

print(feature_cols)

print("\nJumlah fitur:", X.shape[1])

print("\nDistribusi LABEL:")
print(y.value_counts())


# ============================================================
# 8. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 60)
print("PEMBAGIAN DATA")
print("=" * 60)

print("Data training:", len(X_train))
print("Data testing :", len(X_test))


# ============================================================
# 9. RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ============================================================
# 10. TRAINING
# ============================================================

print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)


# ============================================================
# 11. PREDIKSI
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 12. AKURASI
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("HASIL EVALUASI")
print("=" * 60)

print(
    f"Akurasi Model: {accuracy * 100:.2f}%"
)


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

labels = [
    "RENDAH",
    "SEDANG",
    "TINGGI"
]

print("\n=== CLASSIFICATION REPORT ===")

print(
    classification_report(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0
    )
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\n=== CONFUSION MATRIX ===")
print(cm)


# ============================================================
# 15. SIMPAN MODEL
# ============================================================

model_data = {
    "model": model,
    "feature_cols": feature_cols,
    "random_state": 42,
    "test_size": 0.20
}

joblib.dump(
    model_data,
    MODEL_PATH
)


# ============================================================
# 16. HASIL KLASIFIKASI
# ============================================================

df_hasil = df[
    [
        "No_Responden",
        "Skor_X1",
        "Skor_X2",
        "Skor_X3",
        "Skor_X4",
        "Total_Y",
        "LABEL"
    ]
].copy()

df_hasil["PREDIKSI_RF"] = model.predict(X)

df_hasil["STATUS"] = (
    df_hasil["LABEL"] ==
    df_hasil["PREDIKSI_RF"]
).map({
    True: "Sesuai",
    False: "Tidak Sesuai"
})


# ============================================================
# 17. SIMPAN HASIL EXCEL
# ============================================================

EXCEL_PATH = os.path.join(
    BASE_DIR,
    "Hasil_Klasifikasi_Responden_RF.xlsx"
)

df_hasil.to_excel(
    EXCEL_PATH,
    index=False
)


# ============================================================
# SELESAI
# ============================================================

print("\n" + "=" * 60)
print("TRAINING SELESAI")
print("=" * 60)

print("Model :", MODEL_PATH)
print("Excel :", EXCEL_PATH)

print("\nFitur model:")
for fitur in feature_cols:
    print("-", fitur)

print("\nJumlah fitur:", len(feature_cols))

print(
    f"\nAkurasi final: {accuracy * 100:.2f}%"
)