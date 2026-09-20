import os
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================================
# KONFIGURASI
# ============================================================
BASE = os.path.dirname(os.path.abspath(__file__))

DATA = os.path.join(BASE, 'tabulasi_data_terbaru(1)(1).xlsx')
MODEL = os.path.join(BASE, 'model_random_forest_110_40.pkl')
RESULT = os.path.join(BASE, 'Hasil_RF_110_40.xlsx')

FEATURES = ['Skor_X1', 'Skor_X2', 'Skor_X3', 'Skor_X4']
TARGET = 'LABEL'
LOCATION = 'Jarak_Kota'

# ============================================================
# UTILITAS
# ============================================================
def unique_columns(columns):
    """Membuat nama kolom unik agar df[col] selalu menghasilkan Series."""
    result = []
    counts = {}

    for raw in columns:
        name = str(raw).strip()

        if not name or name.lower() == 'nan':
            name = 'Unnamed'

        count = counts.get(name, 0)

        if count == 0:
            result.append(name)
        else:
            result.append(f'{name}_{count}')

        counts[name] = count + 1

    return result


def normalize_location(value):
    """Menyeragamkan isi Tempat Tinggal."""
    value = str(value).strip().upper()

    mapping = {
        'JAUH': 'JAUH DARI KOTA',
        'JAUH DARI KOTA': 'JAUH DARI KOTA',
        'JAUH DARI  KOTA': 'JAUH DARI KOTA',
        'DEKAT': 'DEKAT KOTA',
        'DEKAT KOTA': 'DEKAT KOTA',
        'DEKAT DARI KOTA': 'DEKAT KOTA',
    }

    return mapping.get(value, value)


def load_data():
    """Membaca dataset utama Excel dengan dua baris header."""

    if not os.path.exists(DATA):
        raise FileNotFoundError(
            f'Dataset tidak ditemukan:\n{DATA}'
        )

    print('Membaca dataset:')
    print(DATA)

    raw = pd.read_excel(
        DATA,
        sheet_name='TABULASI',
        header=[0, 1]
    )

    # --------------------------------------------------------
    # Flatten MultiIndex header
    # --------------------------------------------------------
    columns = []

    for col1, col2 in raw.columns:
        h1 = str(col1).strip()
        h2 = str(col2).strip()

        if h1 == 'RESP':
            name = 'No_Responden'
        elif h1 == 'Tempat Tinggal':
            name = 'Jarak_Kota'
        elif h1 == 'SKOR TOTAL Y':
            name = 'SKOR_TOTAL_Y'
        elif h1 == 'LABEL':
            name = 'LABEL'
        else:
            name = h2

        columns.append(name)

    raw.columns = unique_columns(columns)

    # --------------------------------------------------------
    # Kolom item yang wajib tersedia
    # --------------------------------------------------------
    item_columns = [
        'X1.1', 'X1.2', 'X1.3',
        'X2.1', 'X2.2', 'X2.3',
        'X3.1', 'X3.2', 'X3.3',
        'X4.1', 'X4.2', 'X4.3',
        'Y1', 'Y2', 'Y3', 'Y4'
    ]

    missing = [c for c in item_columns if c not in raw.columns]

    if missing:
        raise ValueError(
            'Kolom dataset tidak lengkap. Kolom yang tidak ditemukan: '
            + ', '.join(missing)
        )

    # --------------------------------------------------------
    # Konversi item menjadi numerik
    # --------------------------------------------------------
    for col in item_columns:
        raw[col] = pd.to_numeric(raw[col], errors='coerce')

    # --------------------------------------------------------
    # Hitung skor X1-X4
    # --------------------------------------------------------
    raw['Skor_X1'] = raw[
        ['X1.1', 'X1.2', 'X1.3']
    ].sum(axis=1)

    raw['Skor_X2'] = raw[
        ['X2.1', 'X2.2', 'X2.3']
    ].sum(axis=1)

    raw['Skor_X3'] = raw[
        ['X3.1', 'X3.2', 'X3.3']
    ].sum(axis=1)

    raw['Skor_X4'] = raw[
        ['X4.1', 'X4.2', 'X4.3']
    ].sum(axis=1)

    # --------------------------------------------------------
    # Hitung Total Y
    # --------------------------------------------------------
    raw['Total_Y'] = raw[
        ['Y1', 'Y2', 'Y3', 'Y4']
    ].sum(axis=1)

    # --------------------------------------------------------
    # Normalisasi lokasi dan label
    # --------------------------------------------------------
    raw['Jarak_Kota'] = raw['Jarak_Kota'].map(normalize_location)
    raw['LABEL'] = raw['LABEL'].astype(str).str.strip().str.upper()

    # --------------------------------------------------------
    # Buang data yang tidak lengkap
    # --------------------------------------------------------
    before = len(raw)

    raw = raw.dropna(
        subset=FEATURES + [TARGET, LOCATION]
    ).reset_index(drop=True)

    after = len(raw)

    if before != after:
        print(f'Data yang dibuang karena kosong: {before - after}')

    return raw


# ============================================================
# PEMBAGIAN 110 TRAINING / 40 TESTING
# ============================================================
def split_110_40(df):
    """
    Split sesuai rancangan penelitian:

    Total       : 150
    Jauh kota   : 88
    Dekat kota  : 62

    Training    : 110
      - Jauh    : 65
      - Dekat   : 45

    Testing     : 40
      - Jauh    : 23
      - Dekat   : 17

    Random state = 42.
    """

    far = df.index[
        df[LOCATION] == 'JAUH DARI KOTA'
    ].tolist()

    near = df.index[
        df[LOCATION] == 'DEKAT KOTA'
    ].tolist()

    if len(df) != 150:
        raise ValueError(
            f'Jumlah data harus 150, tetapi ditemukan {len(df)}.'
        )

    if len(far) != 88:
        raise ValueError(
            f'Data JAUH DARI KOTA harus 88, tetapi ditemukan {len(far)}.'
        )

    if len(near) != 62:
        raise ValueError(
            f'Data DEKAT KOTA harus 62, tetapi ditemukan {len(near)}.'
        )

    rng = np.random.RandomState(42)

    far = np.array(far, dtype=int)
    near = np.array(near, dtype=int)

    rng.shuffle(far)
    rng.shuffle(near)

    train_indices = np.concatenate([
        far[:65],
        near[:45]
    ])

    test_indices = np.concatenate([
        far[65:],
        near[45:]
    ])

    train_indices = sorted(train_indices.tolist())
    test_indices = sorted(test_indices.tolist())

    train = df.loc[train_indices].copy()
    test = df.loc[test_indices].copy()

    return train, test


# ============================================================
# CLASSIFICATION REPORT YANG AMAN UNTUK PANDAS
# ============================================================
def make_classification_report_df(y_true, y_pred, classes):
    """
    Membuat DataFrame Classification Report tanpa error:

        AttributeError: 'float' object has no attribute 'items'

    Penyebab error lama:
    classification_report output_dict=True memiliki key 'accuracy'
    yang nilainya FLOAT, sedangkan class lainnya berupa DICT.
    """

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(classes),
        output_dict=True,
        zero_division=0
    )

    rows = []

    for label, value in report_dict.items():
        # Kelas individual, macro avg, weighted avg
        if isinstance(value, dict):
            rows.append({
                'Kelas': label,
                'Precision': float(value.get('precision', 0)),
                'Recall': float(value.get('recall', 0)),
                'F1-Score': float(value.get('f1-score', 0)),
                'Support': int(value.get('support', 0))
            })

        # Accuracy adalah float, bukan dictionary
        else:
            rows.append({
                'Kelas': label,
                'Precision': np.nan,
                'Recall': np.nan,
                'F1-Score': float(value),
                'Support': len(y_true)
            })

    report_df = pd.DataFrame(rows)

    return report_df


# ============================================================
# TRAINING
# ============================================================
print('=' * 70)
print('TRAINING RANDOM FOREST - DATASET 150 / 110 / 40')
print('=' * 70)

# Load data
df = load_data()

print('\nDistribusi lokasi:')
print(df[LOCATION].value_counts())

print('\nDistribusi LABEL:')
print(df[TARGET].value_counts())

# Split
train, test = split_110_40(df)

print('\nPembagian data:')
print(f'Total    : {len(df)}')
print(f'Training : {len(train)}')
print(f'Testing  : {len(test)}')

print('\nLokasi Training:')
print(train[LOCATION].value_counts())

print('\nLokasi Testing:')
print(test[LOCATION].value_counts())

# ------------------------------------------------------------
# Model Random Forest
# ------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

X_train = train[FEATURES].to_numpy()
y_train = train[TARGET].to_numpy()

X_test = test[FEATURES].to_numpy()
y_test = test[TARGET].to_numpy()

print('\nMelatih Random Forest 100 tree...')

model.fit(X_train, y_train)

pred = model.predict(X_test)

# ------------------------------------------------------------
# Evaluasi
# ------------------------------------------------------------
accuracy = accuracy_score(y_test, pred)
classes = list(model.classes_)

cm = confusion_matrix(
    y_test,
    pred,
    labels=classes
)

report = make_classification_report_df(
    y_test,
    pred,
    classes
)

# ------------------------------------------------------------
# Data hasil prediksi
# ------------------------------------------------------------
result_test = test.copy()
result_test['Prediksi_RF'] = pred
result_test['Benar'] = np.where(
    result_test[TARGET].to_numpy() == pred,
    'BENAR',
    'SALAH'
)

# ------------------------------------------------------------
# Feature Importance
# ------------------------------------------------------------
feature_importance = pd.DataFrame({
    'Fitur': FEATURES,
    'Importance': model.feature_importances_
}).sort_values(
    'Importance',
    ascending=False
).reset_index(drop=True)

# ------------------------------------------------------------
# Simpan model
# ------------------------------------------------------------
joblib.dump(model, MODEL)

# ------------------------------------------------------------
# Simpan Excel
# ------------------------------------------------------------
cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes
)

cm_df.index.name = 'Aktual / Prediksi'

# Tambahkan ringkasan lokasi
location_summary = pd.DataFrame({
    'Lokasi': [
        'JAUH DARI KOTA',
        'DEKAT KOTA'
    ],
    'Total': [
        int((df[LOCATION] == 'JAUH DARI KOTA').sum()),
        int((df[LOCATION] == 'DEKAT KOTA').sum())
    ],
    'Training': [
        int((train[LOCATION] == 'JAUH DARI KOTA').sum()),
        int((train[LOCATION] == 'DEKAT KOTA').sum())
    ],
    'Testing': [
        int((test[LOCATION] == 'JAUH DARI KOTA').sum()),
        int((test[LOCATION] == 'DEKAT KOTA').sum())
    ]
})

with pd.ExcelWriter(
    RESULT,
    engine='openpyxl'
) as writer:
    df.to_excel(
        writer,
        sheet_name='Data',
        index=False
    )

    train.to_excel(
        writer,
        sheet_name='Data Latih',
        index=False
    )

    result_test.to_excel(
        writer,
        sheet_name='Data Uji',
        index=False
    )

    cm_df.to_excel(
        writer,
        sheet_name='Confusion Matrix'
    )

    report.to_excel(
        writer,
        sheet_name='Classification Report',
        index=False
    )

    feature_importance.to_excel(
        writer,
        sheet_name='Feature Importance',
        index=False
    )

    location_summary.to_excel(
        writer,
        sheet_name='Pembagian Lokasi',
        index=False
    )

# ============================================================
# OUTPUT
# ============================================================
print('\n' + '=' * 70)
print('SELESAI')
print('=' * 70)
print(f'Data       : {len(df)}')
print(f'Training   : {len(train)}')
print(f'Testing    : {len(test)}')
print(f'Jumlah Tree: {model.n_estimators}')
print(f'Akurasi    : {accuracy * 100:.2f}%')
print(f'Model      : {MODEL}')
print(f'Excel      : {RESULT}')
print('=' * 70)
