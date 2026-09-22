import os
import io
import json
import uuid
import base64
import warnings

warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree

# ============================================================
# APP / PATH
# ============================================================
app = Flask(__name__)
app.secret_key = 'skripsi_kecanduan_balita_secret_key_2026'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ORIGINAL_DATASET = os.path.join(BASE_DIR, 'tabulasi_data_terbaru(1)(1).xlsx')
ORIGINAL_MODEL = os.path.join(BASE_DIR, 'model_random_forest_110_40.pkl')
ORIGINAL_EXCEL = os.path.join(BASE_DIR, 'Hasil_RF_110_40.xlsx')

DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
EXCEL_DIR = os.path.join(BASE_DIR, 'hasil_excel')
REGISTRY_PATH = os.path.join(BASE_DIR, 'dataset_registry.json')

for folder in (DATASET_DIR, MODEL_DIR, EXCEL_DIR):
    os.makedirs(folder, exist_ok=True)

FEATURE_COLS = ['Skor_X1', 'Skor_X2', 'Skor_X3', 'Skor_X4']
TARGET = 'LABEL'
LOCATION = 'Jarak_Kota'
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin123'

# ============================================================
# UTILITAS
# ============================================================
def load_registry():
    if not os.path.exists(REGISTRY_PATH):
        return []
    try:
        with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_registry(data):
    with open(REGISTRY_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
    buf.seek(0)
    result = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return result


def unique_columns(columns):
    """Pastikan nama kolom unik sehingga df[col] selalu Series."""
    result = []
    counts = {}
    for raw in columns:
        name = str(raw).strip()
        if name == '' or name.lower() == 'nan':
            name = 'Unnamed'
        count = counts.get(name, 0)
        if count == 0:
            result.append(name)
        else:
            result.append(f'{name}_{count}')
        counts[name] = count + 1
    return result


def as_series(df, col):
    """Ambil kolom sebagai Series walaupun file memiliki kolom duplikat."""
    value = df.loc[:, col]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return value

# ============================================================
# PEMBACA EXCEL / CSV
# ============================================================
def read_excel_dataset(path):
    try:
        xl = pd.ExcelFile(path)
        sheets = xl.sheet_names
    except Exception as e:
        raise ValueError(f'Excel tidak dapat dibaca: {e}')

    if 'TABULASI' in sheets:
        raw = pd.read_excel(path, sheet_name='TABULASI', header=[0, 1])
        flattened = []
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
            flattened.append(name)
        raw.columns = unique_columns(flattened)
        return raw

    # Dataset upload biasa: coba header normal.
    return pd.read_excel(path, sheet_name=0, header=0)


def read_csv_dataset(path):
    df = pd.read_csv(path)
    rename = {
        'Unnamed: 0': 'No_Responden',
        'Tempat Tinggal': 'Jarak_Kota',
        'Unnamed: 21': 'SKOR_TOTAL_Y',
        'Unnamed: 22': 'LABEL',
        'Unnamed: 23': 'LABEL',
        'TOTAL X1': 'TOTAL_X1',
        'TOTAL X2': 'TOTAL_X2',
        'TOTAL X3': 'TOTAL_X3',
        'TOTAL X4': 'TOTAL_X4',
    }
    df = df.rename(columns=rename)
    return df


def prepare_dataset(df):
    df = df.copy()
    df.columns = unique_columns(df.columns)

    # Normalisasi nama kolom yang mungkin muncul pada upload.
    aliases = {
        'Tempat Tinggal': 'Jarak_Kota',
        'Tempat_Tinggal': 'Jarak_Kota',
        'Jarak Kota': 'Jarak_Kota',
        'Lokasi': 'Jarak_Kota',
        'Unnamed: 21': 'SKOR_TOTAL_Y',
        'Unnamed: 22': 'LABEL',
        'Unnamed: 23': 'LABEL',
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Beberapa Excel bisa membawa header sebagai nama TOTAL X1, dst.
    total_aliases = {
        'TOTAL X1': 'Skor_X1',
        'TOTAL_X1': 'Skor_X1',
        'TOTAL X2': 'Skor_X2',
        'TOTAL_X2': 'Skor_X2',
        'TOTAL X3': 'Skor_X3',
        'TOTAL_X3': 'Skor_X3',
        'TOTAL X4': 'Skor_X4',
        'TOTAL_X4': 'Skor_X4',
    }
    for old, new in total_aliases.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Jika masih ada nama duplikat setelah rename, buat unik.
    df.columns = unique_columns(df.columns)

    if 'Jarak_Kota' not in df.columns:
        raise ValueError('Kolom Tempat Tinggal/Jarak Kota tidak ditemukan.')
    if TARGET not in df.columns:
        raise ValueError('Kolom LABEL tidak ditemukan.')

    # Lokasi.
    df['Jarak_Kota'] = (
        as_series(df, 'Jarak_Kota')
        .astype(str).str.strip().str.upper()
        .replace({
            'JAUH': 'JAUH DARI KOTA',
            'JAUH DARI KOTA': 'JAUH DARI KOTA',
            'DEKAT': 'DEKAT KOTA',
            'DEKAT KOTA': 'DEKAT KOTA',
        })
    )

    # LABEL.
    df[TARGET] = as_series(df, TARGET).astype(str).str.strip().str.upper()

    # Skor X1-X4 selalu dihitung dari item X1.1-X4.3 jika item tersedia.
    groups = {
        'Skor_X1': ['X1.1', 'X1.2', 'X1.3'],
        'Skor_X2': ['X2.1', 'X2.2', 'X2.3'],
        'Skor_X3': ['X3.1', 'X3.2', 'X3.3'],
        'Skor_X4': ['X4.1', 'X4.2', 'X4.3'],
    }
    for score_col, item_cols in groups.items():
        if all(c in df.columns for c in item_cols):
            for c in item_cols:
                df[c] = pd.to_numeric(as_series(df, c), errors='coerce')
            df[score_col] = df[item_cols].sum(axis=1)
        elif score_col in df.columns:
            df[score_col] = pd.to_numeric(as_series(df, score_col), errors='coerce')
        else:
            raise ValueError(f'Kolom {score_col} atau item pembentuknya tidak ditemukan.')

    # Total Y dari Y1-Y4 jika tersedia.
    y_cols = ['Y1', 'Y2', 'Y3', 'Y4']
    if all(c in df.columns for c in y_cols):
        for c in y_cols:
            df[c] = pd.to_numeric(as_series(df, c), errors='coerce')
        df['Total_Y'] = df[y_cols].sum(axis=1)
    elif 'SKOR_TOTAL_Y' in df.columns:
        df['Total_Y'] = pd.to_numeric(as_series(df, 'SKOR_TOTAL_Y'), errors='coerce')

    # Fitur utama.
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(as_series(df, c), errors='coerce')

    # ID responden.
    if 'No_Responden' in df.columns:
        df['No_Responden'] = pd.to_numeric(as_series(df, 'No_Responden'), errors='coerce')
    else:
        df.insert(0, 'No_Responden', np.arange(1, len(df) + 1))

    required = ['No_Responden', LOCATION] + FEATURE_COLS + [TARGET]
    df = df.dropna(subset=required).reset_index(drop=True)

    if df.empty:
        raise ValueError('Tidak ada data valid setelah pembersihan.')
    if df[TARGET].nunique() < 2:
        raise ValueError('LABEL minimal harus memiliki 2 kategori.')

    return df


def load_dataset_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls'):
        df = read_excel_dataset(path)
    elif ext == '.csv':
        df = read_csv_dataset(path)
    else:
        raise ValueError('Format file harus .xlsx, .xls, atau .csv.')
    return prepare_dataset(df)

# ============================================================
# MODEL
# ============================================================
def load_model(path):
    if not os.path.exists(path):
        return None
    obj = joblib.load(path)
    if isinstance(obj, dict) and 'model' in obj:
        return obj['model']
    return obj


def split_110_40(df):
    # Dataset penelitian utama: tepat 65 jauh + 45 dekat untuk training.
    if len(df) == 150:
        far_idx = df.index[df[LOCATION] == 'JAUH DARI KOTA'].tolist()
        near_idx = df.index[df[LOCATION] == 'DEKAT KOTA'].tolist()
        if len(far_idx) == 88 and len(near_idx) == 62:
            rng = np.random.RandomState(42)
            far_idx = np.array(far_idx)
            near_idx = np.array(near_idx)
            rng.shuffle(far_idx)
            rng.shuffle(near_idx)
            train_idx = list(far_idx[:65]) + list(near_idx[:45])
            test_idx = list(far_idx[65:]) + list(near_idx[45:])
            return df.loc[sorted(train_idx)].copy(), df.loc[sorted(test_idx)].copy()

    # Dataset tambahan: 80/20 dengan stratifikasi LABEL.
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=0.20,
            random_state=42,
            stratify=df[TARGET]
        )
    except ValueError:
        train_df, test_df = train_test_split(
            df,
            test_size=0.20,
            random_state=42
        )
    return train_df.copy(), test_df.copy()


def evaluate_model(model, train_df, test_df):
    X_train = train_df.loc[:, FEATURE_COLS].copy()
    y_train = train_df.loc[:, TARGET].copy()
    X_test = test_df.loc[:, FEATURE_COLS].copy()
    y_test = test_df.loc[:, TARGET].copy()

    # Pastikan semua input sklearn 2D dan numerik.
    X_train = X_train.apply(pd.to_numeric, errors='coerce').astype(float)
    X_test = X_test.apply(pd.to_numeric, errors='coerce').astype(float)
    y_train = y_train.astype(str)
    y_test = y_test.astype(str)

    y_pred = model.predict(X_test)
    labels = [str(x) for x in getattr(model, 'classes_', sorted(y_test.unique().tolist()))]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    hasil_uji = test_df.copy()
    hasil_uji['Prediksi_RF'] = y_pred

    return {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'cm': np.asarray(cm, dtype=int),
        'report': report,
        'labels': labels,
        'hasil_uji': hasil_uji,
        'y_test': np.asarray(y_test),
        'y_pred': np.asarray(y_pred),
    }


def classification_report_dataframe(report, accuracy=None):
    """Convert sklearn classification_report(output_dict=True) safely to DataFrame.

    sklearn returns metric dictionaries for classes/macro avg/weighted avg, but
    the ``accuracy`` entry is a scalar float. Passing the mixed dictionary
    directly to DataFrame.from_dict(..., orient="index") raises:
    AttributeError: 'float' object has no attribute 'items'.
    """
    rows = {}
    for key, value in report.items():
        if isinstance(value, dict):
            rows[str(key)] = value.copy()

    df_report = pd.DataFrame.from_dict(rows, orient='index')

    if accuracy is None and 'accuracy' in report:
        try:
            accuracy = float(report['accuracy'])
        except (TypeError, ValueError):
            accuracy = None

    if accuracy is not None:
        # Keep the same four columns as sklearn where possible.
        if 'precision' not in df_report.columns:
            df_report['precision'] = np.nan
        if 'recall' not in df_report.columns:
            df_report['recall'] = np.nan
        if 'f1-score' not in df_report.columns:
            df_report['f1-score'] = np.nan
        if 'support' not in df_report.columns:
            df_report['support'] = np.nan
        df_report.loc['accuracy', 'precision'] = float(accuracy)
        df_report.loc['accuracy', 'recall'] = float(accuracy)
        df_report.loc['accuracy', 'f1-score'] = float(accuracy)
        df_report.loc['accuracy', 'support'] = np.nan

    return df_report.round(3)

# ============================================================
# ACTIVE DATASET
# ============================================================
def get_active_dataset():
    active_id = session.get('active_dataset', 'original')

    if active_id == 'original':
        df = load_dataset_file(ORIGINAL_DATASET)
        model = load_model(ORIGINAL_MODEL)
        if df is None:
            raise FileNotFoundError(f'Dataset tidak ditemukan: {ORIGINAL_DATASET}')
        if model is None:
            raise FileNotFoundError(f'Model tidak ditemukan: {ORIGINAL_MODEL}')
        return {
            'id': 'original',
            'name': 'Data Penelitian Terbaru',
            'file': os.path.basename(ORIGINAL_DATASET),
            'model_file': os.path.basename(ORIGINAL_MODEL),
            'excel_file': os.path.basename(ORIGINAL_EXCEL),
            'path': ORIGINAL_DATASET,
            'model_path': ORIGINAL_MODEL,
            'excel_path': ORIGINAL_EXCEL,
            'df': df,
            'model': model,
        }

    info = next((x for x in load_registry() if x.get('id') == active_id), None)
    if info is None:
        session['active_dataset'] = 'original'
        return get_active_dataset()

    dataset_path = os.path.join(DATASET_DIR, info['dataset_file'])
    model_path = os.path.join(MODEL_DIR, info['model_file'])
    excel_path = os.path.join(EXCEL_DIR, info['excel_file'])

    try:
        df = load_dataset_file(dataset_path)
        model = load_model(model_path)
        if df is None or model is None:
            raise ValueError('Dataset/model tidak ditemukan.')
    except Exception:
        session['active_dataset'] = 'original'
        return get_active_dataset()

    return {
        'id': info['id'],
        'name': info.get('name', info['dataset_file']),
        'file': info['dataset_file'],
        'model_file': info['model_file'],
        'excel_file': info['excel_file'],
        'path': dataset_path,
        'model_path': model_path,
        'excel_path': excel_path,
        'df': df,
        'model': model,
    }

# ============================================================
# CONTEXT / LOGIN
# ============================================================
@app.context_processor
def inject_global():
    return {
        'active_dataset_id': session.get('active_dataset', 'original'),
        'logged_in': session.get('logged_in', False),
    }

# ============================================================
# DASHBOARD
# ============================================================
@app.route('/')
@app.route('/dashboard')
def dashboard():
    dataset = get_active_dataset()
    df, model = dataset['df'], dataset['model']
    train_df, test_df = split_110_40(df)
    result = evaluate_model(model, train_df, test_df)

    accuracy = result['accuracy'] * 100.0
    labels, cm = result['labels'], result['cm']

    report_df = classification_report_dataframe(result['report'], result['accuracy'])
    report_df.index.name = 'Kelas'

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.imshow(cm)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha='center', va='center', fontsize=12)
    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Aktual')
    ax.set_title('Confusion Matrix')
    img_cm = fig_to_base64(fig)

    actual_counts = [int(np.sum(result['y_test'] == label)) for label in labels]
    pred_counts = [int(np.sum(result['y_pred'] == label)) for label in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(x - 0.18, actual_counts, 0.36, label='Aktual')
    ax.bar(x + 0.18, pred_counts, 0.36, label='Prediksi')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Jumlah Responden')
    ax.set_title('Aktual vs Prediksi')
    ax.legend()
    img_bar = fig_to_base64(fig)

    far_total = int((df[LOCATION] == 'JAUH DARI KOTA').sum())
    near_total = int((df[LOCATION] == 'DEKAT KOTA').sum())
    far_high = int(((df[LOCATION] == 'JAUH DARI KOTA') & (df[TARGET] == 'TINGGI')).sum())
    near_high = int(((df[LOCATION] == 'DEKAT KOTA') & (df[TARGET] == 'TINGGI')).sum())
    far_pct = far_high / far_total * 100 if far_total else 0
    near_pct = near_high / near_total * 100 if near_total else 0

    loc_test = test_df[LOCATION].value_counts().reindex(
        ['JAUH DARI KOTA', 'DEKAT KOTA'], fill_value=0
    ).rename_axis('Jarak Kota').reset_index(name='Jumlah')

    return render_template(
        'dashboard.html',
        accuracy=float(accuracy),
        report_table=report_df.to_html(classes='data-table', border=0),
        img_cm=img_cm,
        img_bar=img_bar,
        model_name=dataset['name'],
        total_data=len(df),
        training_data=len(train_df),
        testing_data=len(test_df),
        jumlah_tree=int(getattr(model, 'n_estimators', 100)),
        near_high=near_high,
        far_high=far_high,
        near_pct=float(near_pct),
        far_pct=float(far_pct),
        near_total=near_total,
        far_total=far_total,
        location_test=loc_test.to_html(classes='data-table', index=False, border=0),
    )

# ============================================================
# TABULASI
# ============================================================
@app.route('/tabulasi')
def tabulasi():
    dataset = get_active_dataset()
    df = dataset['df']
    return render_template(
        'tabulasi.html',
        table_html=df.to_html(classes='data-table compact', index=False, border=0),
        total_rows=len(df),
        model_name=dataset['name']
    )

# ============================================================
# SKOR TARGET
# ============================================================
@app.route('/skor-target')
def skor_target():
    dataset = get_active_dataset()
    df = dataset['df']
    cols = ['No_Responden', 'Jarak_Kota', 'Skor_X1', 'Skor_X2', 'Skor_X3', 'Skor_X4', 'Total_Y', 'LABEL']
    cols = [c for c in cols if c in df.columns]
    score_df = df.loc[:, cols].copy()
    target_df = df[TARGET].value_counts().rename_axis('LABEL').reset_index(name='Jumlah')
    model = dataset['model']
    return render_template(
        'skor_target.html',
        skor_table=score_df.to_html(classes='data-table compact', index=False, border=0),
        target_table=target_df.to_html(classes='data-table compact', index=False, border=0),
        model_info={
            'nama_file': dataset['model_file'],
            'tipe_model': type(model).__name__,
            'jumlah_pohon': int(getattr(model, 'n_estimators', 100)),
            'random_state': getattr(model, 'random_state', 42),
        }
    )

# ============================================================
# ANALISIS LOKASI
# ============================================================
@app.route('/analisis-lokasi')
def analisis_lokasi():
    dataset = get_active_dataset()
    df, model = dataset['df'], dataset['model']
    train_df, test_df = split_110_40(df)
    result = evaluate_model(model, train_df, test_df)
    hasil_uji = result['hasil_uji']

    def counts(frame):
        return frame[LOCATION].value_counts().reindex(
            ['JAUH DARI KOTA', 'DEKAT KOTA'], fill_value=0
        ).to_dict()

    label_table = hasil_uji.groupby(LOCATION)[TARGET].value_counts().unstack(fill_value=0)
    pred_table = hasil_uji.groupby(LOCATION)['Prediksi_RF'].value_counts().unstack(fill_value=0)

    tinggi_aktual = hasil_uji[hasil_uji[TARGET] == 'TINGGI'].groupby(LOCATION).size().to_dict()
    tinggi_prediksi = hasil_uji[hasil_uji['Prediksi_RF'] == 'TINGGI'].groupby(LOCATION).size().to_dict()

    return render_template(
        'analisis_lokasi.html',
        total_lokasi=counts(df),
        train_lokasi=counts(train_df),
        test_lokasi=counts(test_df),
        tabel_label=label_table.to_html(classes='data-table compact', border=0),
        tabel_prediksi=pred_table.to_html(classes='data-table compact', border=0),
        kecanduan=tinggi_aktual,
        tinggi_prediksi=tinggi_prediksi,
        hasil_uji=hasil_uji.to_dict(orient='records')
    )

# ============================================================
# EXCEL
# ============================================================
@app.route('/excel')
def excel_view():
    dataset = get_active_dataset()
    path = dataset['excel_path']
    if not os.path.exists(path):
        return render_template('excel_view.html', excel_table='<div class="empty">File hasil Excel belum tersedia.</div>')

    try:
        xls = pd.ExcelFile(path)
        sections = []
        for sheet in xls.sheet_names:
            data = pd.read_excel(path, sheet_name=sheet)
            sections.append(
                f'<div class="sheet-card"><h3>{sheet}</h3>'
                + data.to_html(classes='data-table compact', index=False, border=0)
                + '</div>'
            )
        html = ''.join(sections)
    except Exception as e:
        html = f'<div class="alert danger">Gagal membaca Excel: {e}</div>'
    return render_template('excel_view.html', excel_table=html)

# ============================================================
# DIAGNOSA
# ============================================================
@app.route('/diagnosa', methods=['GET', 'POST'])
def diagnosa():
    dataset = get_active_dataset()
    model = dataset['model']
    hasil = None
    probabilitas = None
    values = {'x1': '', 'x2': '', 'x3': '', 'x4': ''}

    if request.method == 'POST':
        try:
            values = {k: request.form.get(k, '') for k in values}
            arr = [[float(values['x1']), float(values['x2']), float(values['x3']), float(values['x4'])]]
            X = pd.DataFrame(arr, columns=FEATURE_COLS, dtype=float)
            hasil = str(model.predict(X)[0])
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)[0]
                probabilitas = {str(c): f'{float(p) * 100:.2f}%' for c, p in zip(model.classes_, probs)}
        except Exception as e:
            flash(f'Input tidak valid: {e}', 'danger')
    return render_template('diagnosa.html', hasil=hasil, probabilitas=probabilitas, values=values)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
@app.route('/feature-importance')
def feature_importance():
    dataset = get_active_dataset()
    model = dataset['model']
    importances = np.asarray(getattr(model, 'feature_importances_', np.zeros(len(FEATURE_COLS))), dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    order = np.argsort(importances)
    ax.barh(np.array(FEATURE_COLS)[order], importances[order])
    ax.set_xlabel('Nilai Importance')
    ax.set_title('Feature Importance Random Forest')
    for y, v in enumerate(importances[order]):
        ax.text(v + 0.005, y, f'{v:.3f}', va='center')
    img_fi = fig_to_base64(fig)
    return render_template('feature_importance.html', img_fi=img_fi)

# ============================================================
# DECISION TREE
# ============================================================
@app.route('/decision-tree')
def decision_tree():
    dataset = get_active_dataset()
    model = dataset['model']
    total_trees = int(getattr(model, 'n_estimators', len(getattr(model, 'estimators_', []))))
    total_trees = max(total_trees, 1)

    try:
        tree_number = int(request.args.get('tree', 1))
    except ValueError:
        tree_number = 1
    tree_number = max(1, min(tree_number, total_trees))
    tree = model.estimators_[tree_number - 1]

    fig, ax = plt.subplots(figsize=(18, 10))
    plot_tree(
        tree,
        feature_names=FEATURE_COLS,
        class_names=[str(x) for x in model.classes_],
        filled=True,
        rounded=True,
        max_depth=3,
        ax=ax
    )
    ax.set_title(f'Decision Tree #{tree_number} dari Random Forest')
    img_tree = fig_to_base64(fig)

    previous_tree = total_trees if tree_number == 1 else tree_number - 1
    next_tree = 1 if tree_number == total_trees else tree_number + 1

    return render_template(
        'decision_tree.html',
        img_tree=img_tree,
        total_trees=total_trees,
        tree_number=tree_number,
        previous_tree=previous_tree,
        next_tree=next_tree
    )

# ============================================================
# LOGIN / ADMIN
# ============================================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            session['admin_nama'] = 'Administrator'
            return redirect(url_for('admin'))
        flash('Username atau password salah.', 'danger')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('dashboard'))


def train_uploaded_dataset(df):
    """Train model upload. Semua input dibuat Series/2D secara eksplisit."""
    train_df, test_df = split_110_40(df)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train = train_df.loc[:, FEATURE_COLS].astype(float).to_numpy()
    y_train = train_df.loc[:, TARGET].astype(str).to_numpy()
    model.fit(X_train, y_train)
    result = evaluate_model(model, train_df, test_df)
    return model, train_df, test_df, result


def save_result_excel(df, train_df, test_df, result, path):
    report_df = classification_report_dataframe(result['report'], result['accuracy'])
    report_df.index.name = 'Kelas'
    cm_df = pd.DataFrame(
        np.asarray(result['cm'], dtype=int),
        index=[str(x) for x in result['labels']],
        columns=[str(x) for x in result['labels']]
    )

    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Data', index=False)
        train_df.to_excel(writer, sheet_name='Data Latih', index=False)
        test_df_with_pred = result['hasil_uji']
        test_df_with_pred.to_excel(writer, sheet_name='Data Uji', index=False)
        cm_df.to_excel(writer, sheet_name='Confusion Matrix')
        report_df.to_excel(writer, sheet_name='Classification Report')


@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if not session.get('logged_in', False):
        return redirect(url_for('login'))

    # =========================================================
    # POST: PILIH DATASET
    # =========================================================
    if request.method == 'POST':
        action = request.form.get('action', '').strip()

        # -----------------------------------------------------
        # PILIH DATASET
        # -----------------------------------------------------
        if action == 'select_dataset':
            dataset_id = request.form.get('dataset_id', 'original').strip()

            # Dataset original selalu valid
            if dataset_id == 'original':
                session['active_dataset'] = 'original'
                flash('Dataset Data Penelitian Terbaru berhasil dipilih.', 'success')
                return redirect(url_for('admin'))

            # Cek registry
            registry = load_registry()

            selected = next(
                (item for item in registry if item.get('id') == dataset_id),
                None
            )

            if selected:
                session['active_dataset'] = dataset_id
                flash(
                    f'Dataset "{selected.get("name", "Dataset")}" berhasil dipilih.',
                    'success'
                )
            else:
                # Jika dataset sudah tidak ada, kembali ke original
                session['active_dataset'] = 'original'
                flash(
                    'Dataset yang dipilih sudah tidak tersedia. '
                    'Sistem kembali menggunakan Data Penelitian Terbaru.',
                    'warning'
                )

            return redirect(url_for('admin'))

        # -----------------------------------------------------
        # UPLOAD DATASET BARU
        # -----------------------------------------------------
        if action == 'upload_dataset':

            file = request.files.get('file_dataset')

            if not file or not file.filename:
                flash('Silakan pilih file terlebih dahulu.', 'danger')
                return redirect(url_for('admin'))

            ext = os.path.splitext(file.filename)[1].lower()

            if ext not in ('.xlsx', '.xls', '.csv'):
                flash(
                    'Format file harus .xlsx, .xls, atau .csv.',
                    'danger'
                )
                return redirect(url_for('admin'))

            dataset_id = 'dataset_' + uuid.uuid4().hex[:8]

            safe_name = secure_filename(file.filename)

            if not safe_name:
                flash('Nama file tidak valid.', 'danger')
                return redirect(url_for('admin'))

            dataset_filename = dataset_id + '_' + safe_name

            dataset_path = os.path.join(
                DATASET_DIR,
                dataset_filename
            )

            model_filename = dataset_id + '_model.pkl'
            model_path = os.path.join(
                MODEL_DIR,
                model_filename
            )

            excel_filename = dataset_id + '_hasil.xlsx'
            excel_path = os.path.join(
                EXCEL_DIR,
                excel_filename
            )

            try:

                # =================================================
                # 1. SIMPAN FILE DATASET
                # =================================================
                file.save(dataset_path)

                # =================================================
                # 2. BACA DATASET
                # =================================================
                df = load_dataset_file(dataset_path)

                if df is None or df.empty:
                    raise ValueError(
                        'Dataset kosong atau tidak dapat dibaca.'
                    )

                if len(df) < 10:
                    raise ValueError(
                        'Dataset minimal memiliki 10 responden '
                        'setelah pembersihan.'
                    )

                # =================================================
                # 3. TRAINING RANDOM FOREST
                # =================================================
                model, train_df, test_df, result = train_uploaded_dataset(df)

                # =================================================
                # 4. SIMPAN MODEL
                # =================================================
                joblib.dump(
                    model,
                    model_path
                )

                # =================================================
                # 5. SIMPAN HASIL EXCEL
                # =================================================
                save_result_excel(
                    df,
                    train_df,
                    test_df,
                    result,
                    excel_path
                )

                # =================================================
                # 6. UPDATE REGISTRY
                # =================================================
                registry = load_registry()

                # Hapus entry dengan ID yang sama jika ada
                registry = [
                    item for item in registry
                    if item.get('id') != dataset_id
                ]

                new_dataset = {
                    'id': dataset_id,
                    'name': safe_name,
                    'dataset_file': dataset_filename,
                    'model_file': model_filename,
                    'excel_file': excel_filename,
                    'total_data': int(len(df)),
                    'training_data': int(len(train_df)),
                    'testing_data': int(len(test_df)),
                    'accuracy': round(
                        float(result['accuracy']) * 100,
                        2
                    ),
                    'jumlah_tree': int(
                        getattr(model, 'n_estimators', 100)
                    )
                }

                registry.append(new_dataset)

                save_registry(registry)

                # =================================================
                # 7. AKTIFKAN DATASET BARU
                # =================================================
                session['active_dataset'] = dataset_id

                flash(
                    f'Dataset "{safe_name}" berhasil diproses. '
                    f'Akurasi data uji: '
                    f'{float(result["accuracy"]) * 100:.2f}%',
                    'success'
                )

            except Exception as e:

                # ---------------------------------------------
                # HAPUS FILE JIKA PROSES GAGAL
                # ---------------------------------------------
                for p in (
                    dataset_path,
                    model_path,
                    excel_path
                ):
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass

                flash(
                    f'Gagal memproses dataset: '
                    f'{type(e).__name__}: {e}',
                    'danger'
                )

            return redirect(url_for('admin'))

    # =========================================================
    # GET: BERSIHKAN REGISTRY DATASET LAMA
    # =========================================================

    registry = load_registry()

    cleaned_registry = []
    seen_ids = set()

    for item in registry:

        dataset_id = item.get('id')

        if not dataset_id:
            continue

        # -----------------------------------------------------
        # Cegah dataset duplicate
        # -----------------------------------------------------
        if dataset_id in seen_ids:
            continue

        # -----------------------------------------------------
        # Pastikan file dataset benar-benar masih ada
        # -----------------------------------------------------
        dataset_file = item.get('dataset_file', '')
        model_file = item.get('model_file', '')
        excel_file = item.get('excel_file', '')

        dataset_path = os.path.join(
            DATASET_DIR,
            dataset_file
        )

        model_path = os.path.join(
            MODEL_DIR,
            model_file
        )

        excel_path = os.path.join(
            EXCEL_DIR,
            excel_file
        )

        # -----------------------------------------------------
        # Jika dataset sudah dihapus dari folder,
        # jangan tampilkan lagi di Dataset Tersedia.
        #
        # Dataset yang benar-benar tersedia harus memiliki:
        # 1. file dataset
        # 2. model
        # 3. hasil Excel
        # -----------------------------------------------------
        if not os.path.isfile(dataset_path):
            continue

        if not os.path.isfile(model_path):
            continue

        if not os.path.isfile(excel_path):
            continue

        seen_ids.add(dataset_id)

        cleaned_registry.append(item)

    # =========================================================
    # SIMPAN REGISTRY YANG SUDAH DIBERSIHKAN
    # =========================================================

    if cleaned_registry != registry:
        save_registry(cleaned_registry)

    registry = cleaned_registry

    # =========================================================
    # PASTIKAN DATASET AKTIF MASIH TERSEDIA
    # =========================================================

    active_id = session.get(
        'active_dataset',
        'original'
    )

    if active_id != 'original':

        active_exists = any(
            item.get('id') == active_id
            for item in registry
        )

        if not active_exists:
            active_id = 'original'
            session['active_dataset'] = 'original'

            flash(
                'Dataset aktif sebelumnya sudah tidak tersedia. '
                'Sistem menggunakan Data Penelitian Terbaru.',
                'warning'
            )

    # =========================================================
    # AMBIL DATASET AKTIF
    # =========================================================

    dataset = get_active_dataset()

    df = dataset['df']
    model = dataset['model']

    # =========================================================
    # SPLIT DATA
    # =========================================================

    train_df, test_df = split_110_40(df)

    # =========================================================
    # EVALUASI MODEL
    # =========================================================

    result = evaluate_model(
        model,
        train_df,
        test_df
    )

    accuracy_percent = float(
        result.get('accuracy', 0)
    ) * 100.0

    jumlah_tree = int(
        getattr(
            model,
            'n_estimators',
            100
        )
    )

    # =========================================================
    # INFORMASI DATASET AKTIF
    # =========================================================

    if active_id == 'original':

        active_name = 'Data Penelitian Terbaru'

        active_file = os.path.basename(
            ORIGINAL_DATASET
        )

        active_model_file = os.path.basename(
            ORIGINAL_MODEL
        ) if 'ORIGINAL_MODEL' in globals() else 'model_random_forest_150_110_40.pkl'

        active_excel_file = (
            'Hasil_RF_110_40.xlsx'
            if os.path.exists(
                os.path.join(
                    BASE_DIR,
                    'Hasil_RF_110_40.xlsx'
                )
            )
            else '-'
        )

    else:

        active_item = next(
            (
                item
                for item in registry
                if item.get('id') == active_id
            ),
            None
        )

        if active_item:

            active_name = active_item.get(
                'name',
                'Dataset'
            )

            active_file = active_item.get(
                'dataset_file',
                '-'
            )

            active_model_file = active_item.get(
                'model_file',
                '-'
            )

            active_excel_file = active_item.get(
                'excel_file',
                '-'
            )

        else:

            # Safety fallback
            active_id = 'original'
            session['active_dataset'] = 'original'

            active_name = 'Data Penelitian Terbaru'

            active_file = os.path.basename(
                ORIGINAL_DATASET
            )

            active_model_file = 'model_random_forest_150_110_40.pkl'

            active_excel_file = 'Hasil_RF_110_40.xlsx'

    # =========================================================
    # LIST DATASET YANG BENAR-BENAR TERSEDIA
    # =========================================================

    dataset_list = []

    # ---------------------------------------------------------
    # DATA PENELITIAN TERBARU / ORIGINAL
    # ---------------------------------------------------------

    dataset_list.append({
        'id': 'original',
        'name': 'Data Penelitian Terbaru',
        'file': os.path.basename(
            ORIGINAL_DATASET
        ),
        'accuracy': round(
            accuracy_percent,
            2
        ),
        'total_data': int(
            len(df)
        ),
        'training_data': int(
            len(train_df)
        ),
        'testing_data': int(
            len(test_df)
        ),
        'jumlah_tree': jumlah_tree,
        'is_active': active_id == 'original'
    })

    # ---------------------------------------------------------
    # DATASET UPLOAD
    # ---------------------------------------------------------

    for item in registry:

        dataset_id = item.get('id')

        if not dataset_id:
            continue

        dataset_list.append({
            'id': dataset_id,
            'name': item.get(
                'name',
                item.get(
                    'dataset_file',
                    'Dataset'
                )
            ),
            'file': item.get(
                'dataset_file',
                ''
            ),
            'accuracy': float(
                item.get(
                    'accuracy',
                    0
                )
            ),
            'total_data': int(
                item.get(
                    'total_data',
                    0
                )
            ),
            'training_data': int(
                item.get(
                    'training_data',
                    0
                )
            ),
            'testing_data': int(
                item.get(
                    'testing_data',
                    0
                )
            ),
            'jumlah_tree': int(
                item.get(
                    'jumlah_tree',
                    100
                )
            ),
            'is_active': active_id == dataset_id
        })

    # =========================================================
    # DATASET AKTIF - INFORMASI LENGKAP
    # =========================================================

    active_dataset_info = {
        'id': active_id,
        'name': active_name,
        'file': active_file,
        'model_file': active_model_file,
        'excel_file': active_excel_file,
        'total_data': int(len(df)),
        'training_data': int(len(train_df)),
        'testing_data': int(len(test_df)),
        'accuracy': round(
            accuracy_percent,
            2
        ),
        'jumlah_tree': jumlah_tree
    }

    # =========================================================
    # RENDER ADMIN
    # =========================================================

    return render_template(
        'admin.html',

        # -------------------------
        # DATASET AKTIF
        # -------------------------
        model_name=active_name,
        dataset_file=active_file,

        active_dataset=active_id,
        active_dataset_info=active_dataset_info,

        # -------------------------
        # STATISTIK
        # -------------------------
        total_data=int(len(df)),
        training_data=int(len(train_df)),
        testing_data=int(len(test_df)),
        jumlah_tree=jumlah_tree,
        total_trees=jumlah_tree,

        accuracy=float(
            accuracy_percent
        ),

        # -------------------------
        # LIST DATASET
        # -------------------------
        dataset_list=dataset_list,

        # -------------------------
        # REGISTRY
        # -------------------------
        registry=registry
    )


# ============================================================
# START
# ============================================================

if __name__ == '__main__':
    app.run(
        debug=True
    )
