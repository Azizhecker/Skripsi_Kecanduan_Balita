import os
import io
import json
import uuid
import base64
import warnings

warnings.filterwarnings("ignore")

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session
)

from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash

import pandas as pd
import numpy as np
import joblib
import mysql.connector

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.tree import plot_tree


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)
app.secret_key = "skripsi_kecanduan_balita_secret_key"


# ============================================================
# PATH PROJECT
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# DATA PENELITIAN ASLI
# JANGAN DITIMPA
# ============================================================

CSV_PATH = os.path.join(
    BASE_DIR,
    "tabulasi data2.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model_random_forest.pkl"
)

EXCEL_PATH = os.path.join(
    BASE_DIR,
    "Hasil_Klasifikasi_Responden_RF.xlsx"
)


# ============================================================
# FOLDER DATASET BARU
# ============================================================

DATASET_DIR = os.path.join(
    BASE_DIR,
    "datasets"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

EXCEL_DIR = os.path.join(
    BASE_DIR,
    "hasil_excel"
)

REGISTRY_PATH = os.path.join(
    BASE_DIR,
    "dataset_registry.json"
)


os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EXCEL_DIR, exist_ok=True)


# ============================================================
# FITUR RANDOM FOREST
# ============================================================

FEATURE_COLS = [
    "Skor_X1",
    "Skor_X2",
    "Skor_X3",
    "Skor_X4"
]


LABELS = [
    "RENDAH",
    "SEDANG",
    "TINGGI"
]


# ============================================================
# DATABASE
# ============================================================

db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "db_kecanduan_balita"
}


def get_db_connection():

    try:
        return mysql.connector.connect(
            **db_config
        )

    except Exception as e:

        print(
            "Database tidak tersedia:",
            e
        )

        return None


# ============================================================
# LOAD MODEL ASLI
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "model_random_forest.pkl tidak ditemukan. "
        "Pastikan file model penelitian asli berada "
        "di folder project."
    )


model_data = joblib.load(
    MODEL_PATH
)


if isinstance(model_data, dict):

    original_model = model_data["model"]

else:

    original_model = model_data


print("=" * 60)
print("MODEL PENELITIAN ASLI")
print("=" * 60)
print(
    "Tipe model   :",
    type(original_model).__name__
)
print(
    "Jumlah tree  :",
    getattr(
        original_model,
        "n_estimators",
        100
    )
)
print("=" * 60)


# ============================================================
# DATASET AKTIF
# ============================================================

active_dataset_id = "original"


# ============================================================
# REGISTRY DATASET
# ============================================================

def load_registry():

    if not os.path.exists(
        REGISTRY_PATH
    ):
        return []

    try:

        with open(
            REGISTRY_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return []


def save_registry(data):

    with open(
        REGISTRY_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


def add_dataset_registry(
    dataset_id,
    dataset_name,
    dataset_file,
    model_file,
    excel_file,
    total_data,
    training_data,
    testing_data,
    accuracy,
    jumlah_tree
):

    registry = load_registry()

    registry.append({

        "id":
            dataset_id,

        "name":
            dataset_name,

        "dataset_file":
            dataset_file,

        "model_file":
            model_file,

        "excel_file":
            excel_file,

        "total_data":
            int(total_data),

        "training_data":
            int(training_data),

        "testing_data":
            int(testing_data),

        "accuracy":
            float(accuracy),

        "jumlah_tree":
            int(jumlah_tree)

    })

    save_registry(
        registry
    )


def get_registry_dataset(
    dataset_id
):

    registry = load_registry()

    for item in registry:

        if item["id"] == dataset_id:

            return item

    return None


# ============================================================
# FIGURE BASE64
# ============================================================

def fig_to_base64(fig):

    image = io.BytesIO()

    fig.savefig(
        image,
        format="png",
        bbox_inches="tight",
        dpi=120
    )

    image.seek(0)

    return base64.b64encode(
        image.getvalue()
    ).decode("utf-8")


# ============================================================
# LOAD DAN PROSES DATASET
# ============================================================

def load_processed_data(
    csv_path
):

    if not os.path.exists(
        csv_path
    ):

        return None, None, None


    df = pd.read_csv(
        csv_path
    )


    # ========================================================
    # DATA BARU
    # SUDAH MEMILIKI Skor_X1-X4 DAN LABEL
    # ========================================================

    if all(
        col in df.columns
        for col in FEATURE_COLS + ["LABEL"]
    ):

        if "No_Responden" not in df.columns:

            df.insert(
                0,
                "No_Responden",
                range(
                    1,
                    len(df) + 1
                )
            )


        for col in FEATURE_COLS:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )


        df["LABEL"] = (
            df["LABEL"]
            .astype(str)
            .str.upper()
            .str.strip()
        )


        df = df.dropna(
            subset=
            FEATURE_COLS + ["LABEL"]
        ).reset_index(
            drop=True
        )


        X = df[
            FEATURE_COLS
        ].copy()

        y = df["LABEL"].copy()


        return df, X, y


    # ========================================================
    # DATA ASLI
    # ========================================================

    df = df.rename(
        columns={

            "Unnamed: 0":
                "No_Responden",

            "TOTAL X1":
                "Skor_X1",

            "TOTAL X2":
                "Skor_X2",

            "TOTAL X3":
                "Skor_X3",

            "TOTAL X4":
                "Skor_X4",

            "Unnamed: 21":
                "SKOR_TOTAL_Y",

            "Unnamed: 22":
                "LABEL"
        }
    )


    # ========================================================
    # JIKA SKOR X BELUM ADA
    # HITUNG DARI ITEM
    # ========================================================

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


    if all(
        col in df.columns
        for col in x1_cols
    ):

        df["Skor_X1"] = (
            df[x1_cols]
            .sum(axis=1)
        )


    if all(
        col in df.columns
        for col in x2_cols
    ):

        df["Skor_X2"] = (
            df[x2_cols]
            .sum(axis=1)
        )


    if all(
        col in df.columns
        for col in x3_cols
    ):

        df["Skor_X3"] = (
            df[x3_cols]
            .sum(axis=1)
        )


    if all(
        col in df.columns
        for col in x4_cols
    ):

        df["Skor_X4"] = (
            df[x4_cols]
            .sum(axis=1)
        )


    if all(
        col in df.columns
        for col in y_cols
    ):

        df["Total_Y"] = (
            df[y_cols]
            .sum(axis=1)
        )


    # ========================================================
    # VALIDASI
    # ========================================================

    required = (
        FEATURE_COLS + ["LABEL"]
    )

    missing = [
        col
        for col in required
        if col not in df.columns
    ]


    if missing:

        raise ValueError(
            "Kolom tidak ditemukan: "
            + ", ".join(missing)
        )


    df = df.dropna(
        subset=required
    ).reset_index(
        drop=True
    )


    for col in FEATURE_COLS:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


    df = df.dropna(
        subset=FEATURE_COLS
    ).reset_index(
        drop=True
    )


    X = df[
        FEATURE_COLS
    ].copy()

    y = (
        df["LABEL"]
        .astype(str)
        .str.upper()
        .str.strip()
    )


    return df, X, y


# ============================================================
# DATASET AKTIF
# ============================================================

def get_active_dataset():

    dataset_id = session.get(
        "active_dataset",
        "original"
    )


    # ========================================================
    # DATA ASLI
    # ========================================================

    if dataset_id == "original":

        df, X, y = (
            load_processed_data(
                CSV_PATH
            )
        )


        return {

            "id":
                "original",

            "name":
                "Data Penelitian Asli",

            "dataset_file":
                "tabulasi data2.csv",

            "model_file":
                "model_random_forest.pkl",

            "excel_file":
                os.path.basename(
                    EXCEL_PATH
                ),

            "csv_path":
                CSV_PATH,

            "model_path":
                MODEL_PATH,

            "excel_path":
                EXCEL_PATH,

            "df":
                df,

            "X":
                X,

            "y":
                y,

            "model":
                original_model
        }


    # ========================================================
    # DATA BARU
    # ========================================================

    info = get_registry_dataset(
        dataset_id
    )


    if info is None:

        session[
            "active_dataset"
        ] = "original"

        return get_active_dataset()


    csv_path = os.path.join(
        DATASET_DIR,
        info["dataset_file"]
    )

    model_path = os.path.join(
        MODEL_DIR,
        info["model_file"]
    )

    excel_path = os.path.join(
        EXCEL_DIR,
        info["excel_file"]
    )


    if not os.path.exists(
        csv_path
    ):

        session[
            "active_dataset"
        ] = "original"

        return get_active_dataset()


    if not os.path.exists(
        model_path
    ):

        session[
            "active_dataset"
        ] = "original"

        return get_active_dataset()


    df, X, y = (
        load_processed_data(
            csv_path
        )
    )


    saved_model = joblib.load(
        model_path
    )


    if isinstance(
        saved_model,
        dict
    ):

        active_model = (
            saved_model["model"]
        )

    else:

        active_model = saved_model


    return {

        "id":
            dataset_id,

        "name":
            info["name"],

        "dataset_file":
            info["dataset_file"],

        "model_file":
            info["model_file"],

        "excel_file":
            info["excel_file"],

        "csv_path":
            csv_path,

        "model_path":
            model_path,

        "excel_path":
            excel_path,

        "df":
            df,

        "X":
            X,

        "y":
            y,

        "model":
            active_model
    }


# ============================================================
# TRAINING DATA BARU
# ============================================================

def train_new_dataset(
    uploaded_file
):

    # --------------------------------------------------------
    # ID UNIK
    # --------------------------------------------------------

    dataset_id = (
        "dataset_"
        + uuid.uuid4().hex[:8]
    )


    original_filename = secure_filename(
        uploaded_file.filename
    )


    dataset_name = os.path.splitext(
        original_filename
    )[0]


    dataset_filename = (
        dataset_id
        + ".csv"
    )


    model_filename = (
        dataset_id
        + "_model.pkl"
    )


    excel_filename = (
        dataset_id
        + "_hasil.xlsx"
    )


    csv_path = os.path.join(
        DATASET_DIR,
        dataset_filename
    )

    model_path = os.path.join(
        MODEL_DIR,
        model_filename
    )

    excel_path = os.path.join(
        EXCEL_DIR,
        excel_filename
    )


    # --------------------------------------------------------
    # SIMPAN CSV
    # --------------------------------------------------------

    uploaded_file.save(
        csv_path
    )


    # --------------------------------------------------------
    # BACA DATA
    # --------------------------------------------------------

    df, X, y = (
        load_processed_data(
            csv_path
        )
    )


    if df is None:

        raise ValueError(
            "Dataset tidak dapat dibaca."
        )


    if len(df) < 10:

        raise ValueError(
            "Data minimal 10 responden."
        )


    if y.nunique() < 2:

        raise ValueError(
            "LABEL minimal harus memiliki "
            "2 kategori."
        )


    # --------------------------------------------------------
    # SPLIT 80:20
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    new_model = RandomForestClassifier(

        n_estimators=100,

        random_state=42

    )


    print("=" * 60)
    print("TRAINING DATASET BARU")
    print("=" * 60)
    print(
        "Dataset      :",
        original_filename
    )
    print(
        "Jumlah data  :",
        len(df)
    )
    print(
        "Training     :",
        len(X_train)
    )
    print(
        "Testing      :",
        len(X_test)
    )
    print(
        "Jumlah tree  :",
        new_model.n_estimators
    )


    new_model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # EVALUASI
    # --------------------------------------------------------

    y_pred = new_model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    labels = list(
        new_model.classes_
    )


    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )


    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )


    # --------------------------------------------------------
    # PREDIKSI SELURUH DATA
    # --------------------------------------------------------

    df_hasil = df.copy()


    df_hasil[
        "PREDIKSI_RF"
    ] = new_model.predict(
        X
    )


    # --------------------------------------------------------
    # SIMPAN MODEL
    # --------------------------------------------------------

    joblib.dump(
        new_model,
        model_path
    )


    # --------------------------------------------------------
    # SIMPAN EXCEL
    # --------------------------------------------------------

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl"
    ) as writer:

        df_hasil.to_excel(
            writer,
            sheet_name="Hasil Prediksi",
            index=False
        )


        pd.DataFrame(
            cm,
            index=labels,
            columns=labels
        ).to_excel(
            writer,
            sheet_name="Confusion Matrix"
        )


        pd.DataFrame(
            report
        ).transpose().to_excel(
            writer,
            sheet_name="Classification Report"
        )


        pd.DataFrame({

            "Keterangan": [

                "Dataset",

                "Jumlah Data",

                "Data Training",

                "Data Testing",

                "Jumlah Tree",

                "Random State",

                "Akurasi"

            ],

            "Nilai": [

                original_filename,

                len(df),

                len(X_train),

                len(X_test),

                new_model.n_estimators,

                new_model.random_state,

                accuracy

            ]

        }).to_excel(
            writer,
            sheet_name="Ringkasan",
            index=False
        )


    # --------------------------------------------------------
    # REGISTRY
    # --------------------------------------------------------

    add_dataset_registry(

        dataset_id=
            dataset_id,

        dataset_name=
            dataset_name,

        dataset_file=
            dataset_filename,

        model_file=
            model_filename,

        excel_file=
            excel_filename,

        total_data=
            len(df),

        training_data=
            len(X_train),

        testing_data=
            len(X_test),

        accuracy=
            accuracy,

        jumlah_tree=
            new_model.n_estimators

    )


    print("=" * 60)
    print("TRAINING SELESAI")
    print("=" * 60)
    print(
        "Akurasi :",
        f"{accuracy * 100:.2f}%"
    )
    print(
        "Model   :",
        model_path
    )
    print(
        "Excel   :",
        excel_path
    )


    return {
        "id":
            dataset_id,

        "accuracy":
            accuracy,

        "total_data":
            len(df),

        "training_data":
            len(X_train),

        "testing_data":
            len(X_test)
    }


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    dataset = (
        get_active_dataset()
    )


    if dataset["df"] is None:

        return (
            "Dataset tidak ditemukan."
        )


    df = dataset["df"]
    X = dataset["X"]
    y = dataset["y"]
    active_model = dataset["model"]


    # --------------------------------------------------------
    # SPLIT UNTUK EVALUASI
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )


    y_pred = active_model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    labels = list(
        active_model.classes_
    )


    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )


    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )


    report_df = (
        pd.DataFrame(report)
        .transpose()
        .round(2)
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    fig1, ax1 = plt.subplots(
        figsize=(5, 4)
    )


    image = ax1.imshow(
        cm
    )


    ax1.set_xticks(
        range(len(labels))
    )

    ax1.set_yticks(
        range(len(labels))
    )

    ax1.set_xticklabels(
        labels
    )

    ax1.set_yticklabels(
        labels
    )


    for i in range(
        len(labels)
    ):

        for j in range(
            len(labels)
        ):

            ax1.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )


    ax1.set_xlabel(
        "Prediksi Model"
    )

    ax1.set_ylabel(
        "Kelas Aktual"
    )

    ax1.set_title(
        "Confusion Matrix"
    )


    img_cm_heatmap = (
        fig_to_base64(fig1)
    )

    plt.close(fig1)


    # --------------------------------------------------------
    # BAR CHART
    # --------------------------------------------------------

    actual_counts = [
        np.sum(
            y_test == label
        )
        for label in labels
    ]


    pred_counts = [
        np.sum(
            y_pred == label
        )
        for label in labels
    ]


    fig2, ax2 = plt.subplots(
        figsize=(6, 4)
    )


    x_indices = np.arange(
        len(labels)
    )


    width = 0.35


    ax2.bar(
        x_indices - width / 2,
        actual_counts,
        width,
        label="Aktual"
    )


    ax2.bar(
        x_indices + width / 2,
        pred_counts,
        width,
        label="Prediksi"
    )


    ax2.set_xticks(
        x_indices
    )

    ax2.set_xticklabels(
        labels
    )

    ax2.set_ylabel(
        "Jumlah Responden"
    )

    ax2.set_title(
        "Perbandingan Aktual vs Prediksi"
    )

    ax2.legend()


    img_cm_bar = (
        fig_to_base64(fig2)
    )

    plt.close(fig2)


    return render_template(

        "dashboard.html",

        accuracy=
            f"{accuracy * 100:.2f}",

        report_table=
            report_df.to_html(
                classes=
                "table table-hover "
                "table-striped "
                "align-middle",
                border=0
            ),

        img_cm_heatmap=
            img_cm_heatmap,

        img_cm_bar=
            img_cm_bar,

        model_name=
            dataset["name"],

        total_data=
            len(df),

        training_data=
            len(X_train),

        testing_data=
            len(X_test),

        jumlah_tree=
            getattr(
                active_model,
                "n_estimators",
                100
            )
    )


# ============================================================
# TABULASI
# ============================================================

@app.route("/tabulasi")
def tabulasi():

    dataset = (
        get_active_dataset()
    )


    df = dataset["df"]


    if df is None:

        return (
            "Dataset tidak ditemukan."
        )


    table_html = df.to_html(

        classes=
        "table table-bordered "
        "table-hover "
        "table-striped "
        "text-center "
        "align-middle",

        index=False

    )


    return render_template(

        "tabulasi.html",

        table_html=
            table_html

    )


# ============================================================
# SKOR TARGET
# ============================================================

@app.route("/skor-target")
def skor_target():

    dataset = (
        get_active_dataset()
    )


    df = dataset["df"]
    active_model = dataset["model"]


    if df is None:

        return (
            "Dataset tidak ditemukan."
        )


    columns = [

        "No_Responden",

        "Skor_X1",

        "Skor_X2",

        "Skor_X3",

        "Skor_X4"

    ]


    if "Total_Y" in df.columns:

        columns.append(
            "Total_Y"
        )


    columns.append(
        "LABEL"
    )


    df_skor = df[
        columns
    ]


    skor_table = (
        df_skor.to_html(

            classes=
            "table table-bordered "
            "table-hover "
            "text-center "
            "align-middle",

            index=False

        )
    )


    target_counts = (
        df["LABEL"]
        .value_counts()
        .reset_index()
    )


    target_counts.columns = [

        "Kategori Target",

        "Jumlah Responden"

    ]


    target_table = (
        target_counts.to_html(

            classes=
            "table table-bordered "
            "text-center "
            "align-middle",

            index=False

        )
    )


    model_info = {

        "nama_file":
            dataset["model_file"],

        "lokasi":
            dataset["model_path"],

        "tipe_model":
            type(active_model).__name__,

        "jumlah_pohon":
            getattr(
                active_model,
                "n_estimators",
                100
            ),

        "random_state":
            getattr(
                active_model,
                "random_state",
                42
            )

    }


    return render_template(

        "skor_target.html",

        skor_table=
            skor_table,

        target_table=
            target_table,

        model_info=
            model_info

    )


# ============================================================
# EXCEL VIEW
# ============================================================

@app.route("/excel-view")
def excel_view():

    dataset = (
        get_active_dataset()
    )


    excel_path = (
        dataset["excel_path"]
    )


    if not os.path.exists(
        excel_path
    ):

        return render_template(

            "excel_view.html",

            excel_table=
            "<p class='text-danger'>"
            "File Excel belum dibuat."
            "</p>"

        )


    try:

        df_excel = pd.read_excel(
            excel_path
        )


        excel_table = (
            df_excel.to_html(

                classes=
                "table table-striped "
                "table-hover "
                "table-bordered "
                "align-middle "
                "text-center",

                index=False

            )
        )


    except Exception as e:

        excel_table = (

            "<p class='text-danger'>"

            f"Gagal membaca Excel: {e}"

            "</p>"

        )


    return render_template(

        "excel_view.html",

        excel_table=
            excel_table

    )


# ============================================================
# DIAGNOSA
# ============================================================

@app.route(
    "/diagnosa",
    methods=["GET", "POST"]
)
def diagnosa():

    hasil = None
    probabilitas = None
    evaluasi = None


    dataset = (
        get_active_dataset()
    )


    active_model = (
        dataset["model"]
    )


    if request.method == "POST":

        try:

            x1 = float(
                request.form["x1"]
            )

            x2 = float(
                request.form["x2"]
            )

            x3 = float(
                request.form["x3"]
            )

            x4 = float(
                request.form["x4"]
            )


            input_data = pd.DataFrame({

                "Skor_X1": [x1],

                "Skor_X2": [x2],

                "Skor_X3": [x3],

                "Skor_X4": [x4]

            })


            input_data = (
                input_data[
                    FEATURE_COLS
                ]
            )


            hasil = active_model.predict(
                input_data
            )[0]


            if hasattr(
                active_model,
                "predict_proba"
            ):

                probs = (
                    active_model
                    .predict_proba(
                        input_data
                    )[0]
                )


                probabilitas = dict(

                    zip(

                        active_model.classes_,

                        [
                            f"{p * 100:.1f}%"
                            for p in probs
                        ]

                    )

                )


            # ------------------------------------------------
            # EVALUASI HASIL
            # ------------------------------------------------

            if hasil == "RENDAH":

                evaluasi = {

                    "status":
                        "Aman / Normal",

                    "badge_color":
                        "success",

                    "penjelasan":
                        "Penggunaan smartphone pada balita masih berada dalam tingkat rendah.",

                    "evaluasi_ortu": [

                        "Tetap pertahankan batasan waktu penggunaan gadget.",

                        "Prioritaskan aktivitas fisik dan interaksi langsung.",

                        "Hindari penggunaan smartphone sebelum tidur."

                    ]

                }


            elif hasil == "SEDANG":

                evaluasi = {

                    "status":
                        "Peringatan / Perlu Pengawasan",

                    "badge_color":
                        "warning",

                    "penjelasan":
                        "Penggunaan smartphone berada pada tingkat sedang sehingga perlu pengawasan orang tua.",

                    "evaluasi_ortu": [

                        "Evaluasi kembali jadwal penggunaan smartphone.",

                        "Tingkatkan aktivitas bersama anak.",

                        "Gantikan sebagian waktu layar dengan permainan edukatif."

                    ]

                }


            else:

                evaluasi = {

                    "status":
                        "Bahaya / Risiko Tinggi",

                    "badge_color":
                        "danger",

                    "penjelasan":
                        "Penggunaan smartphone berada pada tingkat tinggi dan membutuhkan perhatian lebih.",

                    "evaluasi_ortu": [

                        "Kurangi penggunaan smartphone secara bertahap.",

                        "Orang tua perlu konsisten memberikan batasan.",

                        "Perbanyak aktivitas keluarga tanpa layar.",

                        "Jika terdapat perubahan perilaku yang mengkhawatirkan, pertimbangkan konsultasi dengan tenaga profesional."

                    ]

                }


            # ------------------------------------------------
            # SIMPAN RIWAYAT
            # ------------------------------------------------

            conn = (
                get_db_connection()
            )


            if conn:

                try:

                    cursor = conn.cursor()


                    cursor.execute(

                        """
                        INSERT INTO riwayat_prediksi
                        (
                            skor_x1,
                            skor_x2,
                            skor_x3,
                            skor_x4,
                            hasil_prediksi
                        )
                        VALUES
                        (%s, %s, %s, %s, %s)
                        """,

                        (
                            x1,
                            x2,
                            x3,
                            x4,
                            hasil
                        )

                    )


                    conn.commit()

                    cursor.close()

                    conn.close()


                except Exception as e:

                    print(
                        "Gagal menyimpan riwayat:",
                        e
                    )


        except Exception as e:

            flash(
                f"Input tidak valid: {e}",
                "danger"
            )


    return render_template(

        "diagnosa.html",

        hasil=hasil,

        probabilitas=
            probabilitas,

        evaluasi=
            evaluasi

    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

@app.route(
    "/feature-importance"
)
def feature_importance():

    dataset = (
        get_active_dataset()
    )


    active_model = (
        dataset["model"]
    )


    importances = (
        active_model
        .feature_importances_
    )


    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )


    ax.barh(
        FEATURE_COLS,
        importances
    )


    ax.set_title(
        "Feature Importance Random Forest"
    )


    ax.set_xlabel(
        "Nilai Importance"
    )


    ax.set_ylabel(
        "Variabel"
    )


    for i, value in enumerate(
        importances
    ):

        ax.text(
            value + 0.01,
            i,
            f"{value:.3f}",
            va="center"
        )


    plt.tight_layout()


    img_fi = (
        fig_to_base64(fig)
    )


    plt.close(fig)


    return render_template(

        "feature_importance.html",

        img_fi=
            img_fi

    )


# ============================================================
# DECISION TREE
# ============================================================

@app.route(
    "/decision-tree"
)
def decision_tree():

    dataset = (
        get_active_dataset()
    )


    active_model = (
        dataset["model"]
    )


    pohon_pertama = (
        active_model
        .estimators_[0]
    )


    fig, ax = plt.subplots(
        figsize=(14, 8)
    )


    plot_tree(

        pohon_pertama,

        feature_names=
            FEATURE_COLS,

        class_names=[
            str(c)
            for c in
            active_model.classes_
        ],

        filled=True,

        rounded=True,

        max_depth=3,

        ax=ax

    )


    ax.set_title(
        "Decision Tree #1 "
        "dalam Random Forest"
    )


    img_tree = (
        fig_to_base64(fig)
    )


    plt.close(fig)


    return render_template(

        "decision_tree.html",

        img_tree=
            img_tree

    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        username = request.form[
            "username"
        ]

        password = request.form[
            "password"
        ]


        conn = (
            get_db_connection()
        )


        if conn:

            try:

                cursor = conn.cursor(
                    dictionary=True
                )


                cursor.execute(

                    """
                    SELECT *
                    FROM admin
                    WHERE username = %s
                    """,

                    (username,)

                )


                admin_data = (
                    cursor.fetchone()
                )


                cursor.close()

                conn.close()


                if admin_data:

                    db_password = (
                        admin_data.get(
                            "password",
                            ""
                        )
                    )


                    valid = False


                    try:

                        valid = check_password_hash(
                            db_password,
                            password
                        )

                    except Exception:

                        valid = (
                            password ==
                            db_password
                        )


                    if valid:

                        session[
                            "logged_in"
                        ] = True


                        session[
                            "admin_nama"
                        ] = admin_data.get(
                            "nama",
                            username
                        )


                        return redirect(
                            url_for("admin")
                        )


            except Exception as e:

                print(
                    "Login database error:",
                    e
                )


        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        if (
            username == "admin"
            and
            password == "admin123"
        ):

            session[
                "logged_in"
            ] = True


            session[
                "admin_nama"
            ] = "Administrator"


            return redirect(
                url_for("admin")
            )


        flash(
            "Username atau Password salah!",
            "danger"
        )


    return render_template(
        "login.html"
    )


# ============================================================
# ADMIN
# ============================================================

@app.route(
    "/admin",
    methods=["GET", "POST"]
)
def admin():

    if not session.get(
        "logged_in"
    ):

        return redirect(
            url_for("login")
        )


    # ========================================================
    # PILIH DATASET
    # ========================================================

    if request.method == "POST":

        action = request.form.get(
            "action"
        )


        # ----------------------------------------------------
        # PILIH DATASET
        # ----------------------------------------------------

        if action == "select_dataset":

            selected_id = (
                request.form.get(
                    "dataset_id"
                )
            )


            if (
                selected_id == "original"
                or
                get_registry_dataset(
                    selected_id
                )
            ):

                session[
                    "active_dataset"
                ] = selected_id


                flash(
                    "Dataset berhasil dipilih.",
                    "success"
                )


            else:

                flash(
                    "Dataset tidak ditemukan.",
                    "danger"
                )


            return redirect(
                url_for("admin")
            )


        # ----------------------------------------------------
        # UPLOAD DATA BARU
        # ----------------------------------------------------

        if action == "upload_dataset":

            if "file_csv" not in request.files:

                flash(
                    "File CSV tidak ditemukan.",
                    "danger"
                )

                return redirect(
                    url_for("admin")
                )


            file = request.files[
                "file_csv"
            ]


            if not file.filename:

                flash(
                    "Silakan pilih file CSV.",
                    "danger"
                )

                return redirect(
                    url_for("admin")
                )


            if not file.filename.lower().endswith(
                ".csv"
            ):

                flash(
                    "File harus berformat CSV.",
                    "danger"
                )

                return redirect(
                    url_for("admin")
                )


            try:

                result = (
                    train_new_dataset(
                        file
                    )
                )


                # Setelah selesai training,
                # dataset baru langsung aktif.

                session[
                    "active_dataset"
                ] = result["id"]


                flash(

                    "Dataset baru berhasil "
                    "disimpan dan dilatih. "

                    f"Akurasi: "
                    f"{result['accuracy'] * 100:.2f}%",

                    "success"

                )


            except Exception as e:

                print(
                    "TRAINING ERROR:",
                    e
                )


                flash(
                    f"Training gagal: {e}",
                    "danger"
                )


            return redirect(
                url_for("admin")
            )


    # ========================================================
    # DATASET AKTIF
    # ========================================================

    dataset = (
        get_active_dataset()
    )


    df = dataset["df"]
    active_model = dataset["model"]


    # ========================================================
    # HITUNG INFO AKTIF
    # ========================================================

    if df is not None:

        X = dataset["X"]
        y = dataset["y"]


        X_train, X_test, y_train, y_test = (
            train_test_split(

                X,
                y,

                test_size=0.20,

                random_state=42,

                stratify=y

            )
        )


        y_pred = (
            active_model
            .predict(X_test)
        )


        accuracy = (
            accuracy_score(
                y_test,
                y_pred
            )
        )


        total_data = len(df)

        training_data = len(
            X_train
        )

        testing_data = len(
            X_test
        )


    else:

        accuracy = 0

        total_data = 0

        training_data = 0

        testing_data = 0


    # ========================================================
    # DATASET LIST
    # ========================================================

    registry = (
        load_registry()
    )


    dataset_list = [

        {

            "id":
                "original",

            "name":
                "Data Penelitian Asli",

            "file":
                "tabulasi data2.csv",

            "accuracy":
                None,

            "total_data":
                None

        }

    ]


    for item in registry:

        dataset_list.append({

            "id":
                item["id"],

            "name":
                item["name"],

            "file":
                item["dataset_file"],

            "accuracy":
                item["accuracy"],

            "total_data":
                item["total_data"]

        })


    return render_template(

        "admin.html",

        model_name=
            dataset["name"],

        dataset_file=
            dataset["dataset_file"],

        total_data=
            total_data,

        training_data=
            training_data,

        testing_data=
            testing_data,

        jumlah_tree=
            getattr(
                active_model,
                "n_estimators",
                100
            ),

        accuracy=
            f"{accuracy * 100:.2f}",

        dataset_list=
            dataset_list,

        active_dataset=
            dataset["id"]

    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("dashboard")
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )