import os
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_FILE = os.path.join(
    BASE_DIR,
    "tabulasi_data_terbaru(1)(1).xlsx"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "model_random_forest_110_40.pkl"
)

EXCEL_RESULT = os.path.join(
    BASE_DIR,
    "Hasil_RF_110_40.xlsx"
)


FEATURES = [
    "Skor_X1",
    "Skor_X2",
    "Skor_X3",
    "Skor_X4"
]

TARGET = "LABEL"

LOCATION = "Jarak_Kota"


# ============================================================
# MEMBACA EXCEL TERBARU
# ============================================================

def load_dataset():

    print("=" * 70)
    print("MEMBACA DATASET PENELITIAN TERBARU")
    print("=" * 70)

    print(
        "\nFile:",
        os.path.basename(DATASET_FILE)
    )

    if not os.path.exists(DATASET_FILE):

        raise FileNotFoundError(
            f"File tidak ditemukan:\n{DATASET_FILE}"
        )

    # File terbaru mempunyai 2 baris header
    df = pd.read_excel(
        DATASET_FILE,
        sheet_name="TABULASI",
        header=[0, 1]
    )

    # --------------------------------------------------------
    # Gabungkan multi-header menjadi nama kolom sederhana
    # --------------------------------------------------------

    new_columns = []

    for col1, col2 in df.columns:

        col1 = str(col1).strip()
        col2 = str(col2).strip()

        if col1 == "RESP":
            nama = "No_Responden"

        elif col1 == "Tempat Tinggal":
            nama = "Jarak_Kota"

        elif col1 == "SKOR TOTAL Y":
            nama = "SKOR_TOTAL_Y"

        elif col1 == "LABEL":
            nama = "LABEL"

        else:
            nama = col2

        new_columns.append(nama)

    df.columns = new_columns

    # --------------------------------------------------------
    # Hapus kolom duplikat jika ada
    # --------------------------------------------------------

    df = df.loc[
        :,
        ~df.columns.duplicated()
    ]

    # --------------------------------------------------------
    # Normalisasi lokasi
    # --------------------------------------------------------

    df["Jarak_Kota"] = (
        df["Jarak_Kota"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["Jarak_Kota"] = (
        df["Jarak_Kota"]
        .replace({
            "jauh dari kota": "JAUH DARI KOTA",
            "dekat kota": "DEKAT KOTA",
            "jauh": "JAUH DARI KOTA",
            "dekat": "DEKAT KOTA"
        })
    )

    # --------------------------------------------------------
    # Normalisasi LABEL
    # --------------------------------------------------------

    df["LABEL"] = (
        df["LABEL"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Hitung Skor X1-X4 dari item
    # --------------------------------------------------------

    kelompok = {

        "Skor_X1": [
            "X1.1",
            "X1.2",
            "X1.3"
        ],

        "Skor_X2": [
            "X2.1",
            "X2.2",
            "X2.3"
        ],

        "Skor_X3": [
            "X3.1",
            "X3.2",
            "X3.3"
        ],

        "Skor_X4": [
            "X4.1",
            "X4.2",
            "X4.3"
        ]
    }

    for skor, kolom in kelompok.items():

        for c in kolom:

            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )

        df[skor] = (
            df[kolom]
            .sum(axis=1)
        )

    # --------------------------------------------------------
    # Total Y
    # --------------------------------------------------------

    y_columns = [
        "Y1",
        "Y2",
        "Y3",
        "Y4"
    ]

    for c in y_columns:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df["Total_Y"] = (
        df[y_columns]
        .sum(axis=1)
    )

    # --------------------------------------------------------
    # Konversi target dan fitur
    # --------------------------------------------------------

    for col in FEATURES:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Hapus baris tidak valid
    # --------------------------------------------------------

    df = df.dropna(
        subset=
        FEATURES
        +
        [
            TARGET,
            LOCATION
        ]
    )

    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================
# PEMBAGIAN DATA 110 : 40
# ============================================================

def split_data(df):

    print("\n")
    print("=" * 70)
    print("PEMBAGIAN DATA 110 DATA LATIH + 40 DATA UJI")
    print("=" * 70)

    rng = np.random.RandomState(42)

    train_indices = []
    test_indices = []

    # --------------------------------------------------------
    # JAUH DARI KOTA
    # 88 → 65 train + 23 test
    # --------------------------------------------------------

    jauh_indices = df.index[
        df[LOCATION]
        == "JAUH DARI KOTA"
    ].tolist()

    # --------------------------------------------------------
    # DEKAT KOTA
    # 62 → 45 train + 17 test
    # --------------------------------------------------------

    dekat_indices = df.index[
        df[LOCATION]
        == "DEKAT KOTA"
    ].tolist()

    print(
        "\nJumlah awal Jauh dari Kota:",
        len(jauh_indices)
    )

    print(
        "Jumlah awal Dekat Kota:",
        len(dekat_indices)
    )

    # Acak dengan random_state 42
    rng.shuffle(
        jauh_indices
    )

    rng.shuffle(
        dekat_indices
    )

    # --------------------------------------------------------
    # Jauh
    # --------------------------------------------------------

    train_indices.extend(
        jauh_indices[:65]
    )

    test_indices.extend(
        jauh_indices[65:]
    )

    # --------------------------------------------------------
    # Dekat
    # --------------------------------------------------------

    train_indices.extend(
        dekat_indices[:45]
    )

    test_indices.extend(
        dekat_indices[45:]
    )

    # Urutkan index
    train_indices = sorted(
        train_indices
    )

    test_indices = sorted(
        test_indices
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_df = df.loc[
        train_indices
    ].copy()

    test_df = df.loc[
        test_indices
    ].copy()

    print("\nDATA LATIH")

    print(
        train_df[
            LOCATION
        ].value_counts()
    )

    print(
        "\nTotal data latih:",
        len(train_df)
    )

    print("\nDATA UJI")

    print(
        test_df[
            LOCATION
        ].value_counts()
    )

    print(
        "\nTotal data uji:",
        len(test_df)
    )

    return (
        train_df,
        test_df
    )


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

def train_model(
    train_df,
    test_df
):

    print("\n")
    print("=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        TARGET
    ]

    X_test = test_df[
        FEATURES
    ]

    y_test = test_df[
        TARGET
    ]

    # --------------------------------------------------------
    # Random Forest 100 Tree
    # --------------------------------------------------------

    model = RandomForestClassifier(

        n_estimators=100,

        random_state=42

    )

    print(
        "\nJumlah Decision Tree:",
        model.n_estimators
    )

    print(
        "Jumlah fitur:",
        len(FEATURES)
    )

    print(
        "Fitur:",
        FEATURES
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Prediksi
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\n")
    print("=" * 70)
    print("HASIL EVALUASI")
    print("=" * 70)

    print(
        f"\nAccuracy: {accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    labels = list(
        model.classes_
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    print("\nCONFUSION MATRIX")

    print(
        pd.DataFrame(
            cm,
            index=labels,
            columns=labels
        )
    )

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    print(
        "\nCLASSIFICATION REPORT"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Hasil data uji
    # --------------------------------------------------------

    hasil_uji = test_df.copy()

    hasil_uji[
        "Prediksi_RF"
    ] = y_pred

    return (
        model,
        accuracy,
        cm,
        report,
        hasil_uji
    )


# ============================================================
# ANALISIS LOKASI
# ============================================================

def analisis_lokasi(
    df,
    train_df,
    test_df,
    hasil_uji
):

    print("\n")
    print("=" * 70)
    print("ANALISIS JARAK TEMPAT TINGGAL")
    print("=" * 70)

    # --------------------------------------------------------
    # Total 150
    # --------------------------------------------------------

    total_lokasi = (
        df[
            LOCATION
        ]
        .value_counts()
    )

    print(
        "\nTOTAL 150 RESPONDEN:"
    )

    print(
        total_lokasi
    )

    # --------------------------------------------------------
    # Training 110
    # --------------------------------------------------------

    train_lokasi = (
        train_df[
            LOCATION
        ]
        .value_counts()
    )

    print(
        "\nDATA LATIH 110:"
    )

    print(
        train_lokasi
    )

    # --------------------------------------------------------
    # Testing 40
    # --------------------------------------------------------

    test_lokasi = (
        test_df[
            LOCATION
        ]
        .value_counts()
    )

    print(
        "\nDATA UJI 40:"
    )

    print(
        test_lokasi
    )

    # --------------------------------------------------------
    # Label aktual data uji
    # --------------------------------------------------------

    tabel_kecanduan = (
        hasil_uji
        .groupby(
            LOCATION
        )[
            TARGET
        ]
        .value_counts()
        .unstack(
            fill_value=0
        )
    )

    print(
        "\nLABEL AKTUAL DATA UJI:"
    )

    print(
        tabel_kecanduan
    )

    # --------------------------------------------------------
    # Prediksi Random Forest
    # --------------------------------------------------------

    tabel_prediksi = (
        hasil_uji
        .groupby(
            LOCATION
        )[
            "Prediksi_RF"
        ]
        .value_counts()
        .unstack(
            fill_value=0
        )
    )

    print(
        "\nPREDIKSI RANDOM FOREST DATA UJI:"
    )

    print(
        tabel_prediksi
    )

    # --------------------------------------------------------
    # TINGGI
    # --------------------------------------------------------

    tinggi_aktual = (
        hasil_uji[
            hasil_uji[TARGET]
            == "TINGGI"
        ]
        .groupby(
            LOCATION
        )
        .size()
    )

    tinggi_prediksi = (
        hasil_uji[
            hasil_uji[
                "Prediksi_RF"
            ]
            == "TINGGI"
        ]
        .groupby(
            LOCATION
        )
        .size()
    )

    print(
        "\nLABEL TINGGI AKTUAL:"
    )

    print(
        tinggi_aktual
    )

    print(
        "\nPREDIKSI TINGGI:"
    )

    print(
        tinggi_prediksi
    )

    return (
        total_lokasi,
        train_lokasi,
        test_lokasi,
        tabel_kecanduan,
        tabel_prediksi,
        tinggi_aktual,
        tinggi_prediksi
    )


# ============================================================
# SIMPAN EXCEL
# ============================================================

def save_excel(
    df,
    train_df,
    test_df,
    hasil_uji,
    cm,
    report,
    total_lokasi,
    train_lokasi,
    test_lokasi,
    tabel_kecanduan,
    tabel_prediksi
):

    print("\n")
    print("=" * 70)
    print("MENYIMPAN HASIL EXCEL")
    print("=" * 70)

    with pd.ExcelWriter(
        EXCEL_RESULT,
        engine="openpyxl"
    ) as writer:

        # ----------------------------------------------------
        # Data 150
        # ----------------------------------------------------

        df.to_excel(
            writer,
            sheet_name="Data 150",
            index=False
        )

        # ----------------------------------------------------
        # Data Latih
        # ----------------------------------------------------

        train_df.to_excel(
            writer,
            sheet_name="Data Latih 110",
            index=False
        )

        # ----------------------------------------------------
        # Data Uji
        # ----------------------------------------------------

        test_df.to_excel(
            writer,
            sheet_name="Data Uji 40",
            index=False
        )

        # ----------------------------------------------------
        # Hasil Prediksi
        # ----------------------------------------------------

        hasil_uji.to_excel(
            writer,
            sheet_name="Hasil Prediksi",
            index=False
        )

        # ----------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------

        pd.DataFrame(
            cm,
            index=[
                "Aktual RENDAH",
                "Aktual SEDANG",
                "Aktual TINGGI"
            ],
            columns=[
                "Pred RENDAH",
                "Pred SEDANG",
                "Pred TINGGI"
            ]
        ).to_excel(
            writer,
            sheet_name="Confusion Matrix"
        )

        # ----------------------------------------------------
        # Classification Report
        # ----------------------------------------------------

        pd.DataFrame(
            report
        ).transpose().to_excel(
            writer,
            sheet_name="Classification Report"
        )

        # ----------------------------------------------------
        # Total lokasi
        # ----------------------------------------------------

        total_lokasi.to_frame(
            "Jumlah"
        ).to_excel(
            writer,
            sheet_name="Analisis Lokasi"
        )

        # ----------------------------------------------------
        # Training lokasi
        # ----------------------------------------------------

        train_lokasi.to_frame(
            "Jumlah"
        ).to_excel(
            writer,
            sheet_name="Data Latih Lokasi"
        )

        # ----------------------------------------------------
        # Testing lokasi
        # ----------------------------------------------------

        test_lokasi.to_frame(
            "Jumlah"
        ).to_excel(
            writer,
            sheet_name="Data Uji Lokasi"
        )

        # ----------------------------------------------------
        # Kecanduan
        # ----------------------------------------------------

        tabel_kecanduan.to_excel(
            writer,
            sheet_name="Kecanduan Berdasarkan Lokasi"
        )

        # ----------------------------------------------------
        # Prediksi
        # ----------------------------------------------------

        tabel_prediksi.to_excel(
            writer,
            sheet_name="Prediksi Berdasarkan Lokasi"
        )

    print(
        "\nExcel berhasil dibuat:"
    )

    print(
        EXCEL_RESULT
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    df = load_dataset()

    print(
        "\nJumlah responden terbaca:",
        len(df)
    )

    # --------------------------------------------------------
    # Validasi 150 responden
    # --------------------------------------------------------

    if len(df) != 150:

        raise ValueError(
            f"Dataset seharusnya 150 responden, "
            f"tetapi terbaca {len(df)}."
        )

    # --------------------------------------------------------
    # Validasi lokasi
    # --------------------------------------------------------

    jumlah_jauh = (
        (
            df[LOCATION]
            == "JAUH DARI KOTA"
        )
        .sum()
    )

    jumlah_dekat = (
        (
            df[LOCATION]
            == "DEKAT KOTA"
        )
        .sum()
    )

    print(
        "\nJumlah Jauh dari Kota:",
        jumlah_jauh
    )

    print(
        "Jumlah Dekat Kota:",
        jumlah_dekat
    )

    if jumlah_jauh != 88:

        raise ValueError(
            "Jumlah Jauh dari Kota bukan 88."
        )

    if jumlah_dekat != 62:

        raise ValueError(
            "Jumlah Dekat Kota bukan 62."
        )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_df, test_df = split_data(
        df
    )

    # --------------------------------------------------------
    # Pastikan 110 + 40
    # --------------------------------------------------------

    if len(train_df) != 110:

        raise ValueError(
            "Data latih harus 110."
        )

    if len(test_df) != 40:

        raise ValueError(
            "Data uji harus 40."
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    (
        model,
        accuracy,
        cm,
        report,
        hasil_uji
    ) = train_model(
        train_df,
        test_df
    )

    # --------------------------------------------------------
    # Analisis lokasi
    # --------------------------------------------------------

    (
        total_lokasi,
        train_lokasi,
        test_lokasi,
        tabel_kecanduan,
        tabel_prediksi,
        tinggi_aktual,
        tinggi_prediksi
    ) = analisis_lokasi(
        df,
        train_df,
        test_df,
        hasil_uji
    )

    # --------------------------------------------------------
    # Simpan model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        "\nModel berhasil disimpan:"
    )

    print(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Simpan Excel
    # --------------------------------------------------------

    save_excel(
        df,
        train_df,
        test_df,
        hasil_uji,
        cm,
        report,
        total_lokasi,
        train_lokasi,
        test_lokasi,
        tabel_kecanduan,
        tabel_prediksi
    )

    print("\n")
    print("=" * 70)
    print("TRAINING SELESAI")
    print("=" * 70)

    print(
        f"""
Dataset       : 150 responden
Data latih    : 110
Data uji      : 40
Random Forest : 100 Tree
Fitur         : 4

Jauh dari kota : 88
Dekat kota     : 62

Model:
{MODEL_FILE}

Excel:
{EXCEL_RESULT}
"""
    )