import os
import io
import base64
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# IMPORT FLASK
# ============================================================

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash
)


# ============================================================
# IMPORT DATA SCIENCE
# ============================================================

import pandas as pd
import numpy as np
import joblib


# ============================================================
# MYSQL
# ============================================================

import mysql.connector


# ============================================================
# VISUALISASI
# ============================================================

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# MACHINE LEARNING
# ============================================================

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

from sklearn.model_selection import train_test_split

from sklearn.tree import plot_tree


# ============================================================
# FLASK CONFIG
# ============================================================

app = Flask(__name__)

app.secret_key = "skripsi_balita_super_secret_key"


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model_random_forest.pkl"
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "tabulasi data2.csv"
)

EXCEL_PATH = os.path.join(
    BASE_DIR,
    "Hasil_Klasifikasi_Responden_RF.xlsx"
)


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        "model_random_forest.pkl tidak ditemukan. "
        "Jalankan train.py terlebih dahulu."
    )


model_data = joblib.load(
    MODEL_PATH
)


# Model
model = model_data["model"]


# Fitur yang digunakan model
feature_cols = model_data.get(
    "feature_cols",
    [
        "Skor_X1",
        "Skor_X2",
        "Skor_X3",
        "Skor_X4"
    ]
)


print("=" * 60)
print("MODEL RANDOM FOREST BERHASIL DIMUAT")
print("=" * 60)

print("Model :", type(model).__name__)

print("Fitur :")
for fitur in feature_cols:
    print("-", fitur)

print("Jumlah fitur:", len(feature_cols))


# ============================================================
# DATABASE CONFIG
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

    except Exception:

        return None


# ============================================================
# FIGURE → BASE64
# ============================================================

def fig_to_base64(fig):

    img = io.BytesIO()

    fig.savefig(
        img,
        format="png",
        bbox_inches="tight",
        dpi=120
    )

    img.seek(0)

    return base64.b64encode(
        img.getvalue()
    ).decode("utf-8")


# ============================================================
# HELPER:
# LOAD DAN PROCESS DATA
# ============================================================

def load_processed_data():

    if not os.path.exists(CSV_PATH):

        return None, None, None


    # --------------------------------------------------------
    # BACA CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        CSV_PATH
    )


    # --------------------------------------------------------
    # RENAME KOLOM
    # --------------------------------------------------------

    df = df.rename(columns={

        "Unnamed: 0":
            "No_Responden",

        "Unnamed: 21":
            "SKOR_TOTAL_Y",

        "Unnamed: 22":
            "LABEL"
    })


    # --------------------------------------------------------
    # HAPUS DATA LABEL KOSONG
    # --------------------------------------------------------

    df = df.dropna(
        subset=["LABEL"]
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # ITEM X1
    # --------------------------------------------------------

    x1_cols = [
        "X1.1",
        "X1.2",
        "X1.3"
    ]


    # --------------------------------------------------------
    # ITEM X2
    # --------------------------------------------------------

    x2_cols = [
        "X2.1",
        "X2.2",
        "X2.3"
    ]


    # --------------------------------------------------------
    # ITEM X3
    # --------------------------------------------------------

    x3_cols = [
        "X3.1",
        "X3.2",
        "X3.3"
    ]


    # --------------------------------------------------------
    # ITEM X4
    # --------------------------------------------------------

    x4_cols = [
        "X4.1",
        "X4.2",
        "X4.3"
    ]


    # --------------------------------------------------------
    # ITEM Y
    # --------------------------------------------------------

    y_cols = [
        "Y1",
        "Y2",
        "Y3",
        "Y4"
    ]


    # --------------------------------------------------------
    # HITUNG SKOR X1-X4
    # --------------------------------------------------------

    df["Skor_X1"] = (
        df[x1_cols]
        .sum(axis=1)
    )

    df["Skor_X2"] = (
        df[x2_cols]
        .sum(axis=1)
    )

    df["Skor_X3"] = (
        df[x3_cols]
        .sum(axis=1)
    )

    df["Skor_X4"] = (
        df[x4_cols]
        .sum(axis=1)
    )


    # --------------------------------------------------------
    # HITUNG TOTAL Y
    # --------------------------------------------------------

    df["Total_Y"] = (
        df[y_cols]
        .sum(axis=1)
    )


    # --------------------------------------------------------
    # FITUR MODEL
    # --------------------------------------------------------

    X = df[
        [
            "Skor_X1",
            "Skor_X2",
            "Skor_X3",
            "Skor_X4"
        ]
    ]


    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    y = df["LABEL"]


    return df, X, y


# ============================================================
# ROUTE 1
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    df_raw, X, y = (
        load_processed_data()
    )


    if df_raw is None:

        return (
            "File tabulasi data2.csv "
            "tidak ditemukan."
        )


    # --------------------------------------------------------
    # SPLIT HARUS SAMA DENGAN JUPYTER
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
    # PREDIKSI
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    acc = (
        accuracy_score(
            y_test,
            y_pred
        ) * 100
    )


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    labels = [
        "RENDAH",
        "SEDANG",
        "TINGGI"
    ]


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

    cm = confusion_matrix(

        y_test,

        y_pred,

        labels=labels
    )


    # --------------------------------------------------------
    # GRAFIK CONFUSION MATRIX
    # --------------------------------------------------------

    fig1, ax1 = plt.subplots(
        figsize=(5, 4)
    )


    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=labels,

        yticklabels=labels,

        ax=ax1
    )


    ax1.set_title(
        "Confusion Matrix",
        fontsize=12,
        fontweight="bold"
    )

    ax1.set_xlabel(
        "Prediksi Model"
    )

    ax1.set_ylabel(
        "Kelas Aktual"
    )


    img_cm_heatmap = (
        fig_to_base64(fig1)
    )


    plt.close(fig1)


    # --------------------------------------------------------
    # BAR CHART AKTUAL VS PREDIKSI
    # --------------------------------------------------------

    fig2, ax2 = plt.subplots(
        figsize=(6, 4)
    )


    x_indices = np.arange(
        len(labels)
    )


    width = 0.35


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


    ax2.set_title(
        "Perbandingan Aktual vs Prediksi",
        fontsize=12,
        fontweight="bold"
    )


    ax2.set_ylabel(
        "Jumlah Responden"
    )


    ax2.legend()


    img_cm_bar = (
        fig_to_base64(fig2)
    )


    plt.close(fig2)


    # --------------------------------------------------------
    # MODAL
    # --------------------------------------------------------

    show_modal = session.pop(
        "show_modal",
        True
    )


    # --------------------------------------------------------
    # RENDER
    # --------------------------------------------------------

    return render_template(

        "dashboard.html",

        accuracy=f"{acc:.2f}",

        report_table=
        report_df.to_html(
            classes=
            "table table-hover table-striped align-middle",
            border=0
        ),

        img_cm_heatmap=
        img_cm_heatmap,

        img_cm_bar=
        img_cm_bar,

        show_modal=
        show_modal
    )


# ============================================================
# ROUTE 2
# TABULASI
# ============================================================

@app.route("/tabulasi")
def tabulasi():

    if os.path.exists(
        CSV_PATH
    ):

        df = pd.read_csv(
            CSV_PATH
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

    else:

        table_html = (
            "<p class='text-danger'>"
            "File tabulasi data2.csv "
            "tidak ditemukan."
            "</p>"
        )


    return render_template(

        "tabulasi.html",

        table_html=
        table_html
    )


# ============================================================
# ROUTE 3
# SKOR X1-X4 DAN TARGET
# ============================================================

@app.route("/skor-target")
def skor_target():

    df_raw, X, y = (
        load_processed_data()
    )


    if df_raw is None:

        return (
            "Dataset tidak ditemukan."
        )


    # --------------------------------------------------------
    # TABEL SKOR
    # --------------------------------------------------------

    df_skor = df_raw[

        [
            "No_Responden",

            "Skor_X1",

            "Skor_X2",

            "Skor_X3",

            "Skor_X4",

            "Total_Y",

            "LABEL"
        ]
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


    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    target_counts = (
        y.value_counts()
        .to_frame()
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


    # --------------------------------------------------------
    # INFORMASI MODEL
    # --------------------------------------------------------

    model_info = {

        "nama_file":
            "model_random_forest.pkl",

        "lokasi":
            MODEL_PATH,

        "ukuran_file":
            (
                f"{os.path.getsize(MODEL_PATH) / 1024:.2f} KB"
                if os.path.exists(MODEL_PATH)
                else "Tidak ditemukan"
            ),

        "tipe_model":
            type(model).__name__,

        "jumlah_pohon":
            model.n_estimators,

        "random_state":
            model.random_state
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
# ROUTE 4
# HASIL EXCEL
# ============================================================

@app.route("/excel-view")
def excel_view():

    if os.path.exists(
        EXCEL_PATH
    ):

        df_excel = pd.read_excel(
            EXCEL_PATH
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

    else:

        excel_table = (
            "<p class='text-danger'>"
            "Hasil_Klasifikasi_Responden_RF.xlsx "
            "belum dibuat."
            "</p>"
        )


    return render_template(

        "excel_view.html",

        excel_table=
        excel_table
    )


# ============================================================
# ROUTE 5
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


    if request.method == "POST":

        # ----------------------------------------------------
        # INPUT SKOR
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame({

            "Skor_X1": [x1],

            "Skor_X2": [x2],

            "Skor_X3": [x3],

            "Skor_X4": [x4]
        })


        # Pastikan urutan fitur sama
        input_data = input_data[
            feature_cols
        ]


        # ----------------------------------------------------
        # PREDIKSI
        # ----------------------------------------------------

        prediksi = model.predict(
            input_data
        )[0]


        # ----------------------------------------------------
        # PROBABILITAS
        # ----------------------------------------------------

        probs = model.predict_proba(
            input_data
        )[0]


        hasil = prediksi


        probabilitas = dict(
            zip(

                model.classes_,

                [
                    f"{p * 100:.1f}%"
                    for p in probs
                ]
            )
        )


        # ----------------------------------------------------
        # EVALUASI RENDAH
        # ----------------------------------------------------

        if prediksi == "RENDAH":

            evaluasi = {

                "status":
                    "Aman / Normal",

                "badge_color":
                    "success",

                "penjelasan":
                    "Penggunaan smartphone pada balita masih berada dalam tingkat batas wajar dan aman.",

                "evaluasi_ortu": [

                    "Tetap pertahankan batasan waktu penggunaan gadget.",

                    "Prioritaskan aktivitas fisik, interaksi tatap muka, dan permainan.",

                    "Hindari memberikan smartphone saat waktu makan atau sebelum tidur."
                ]
            }


        # ----------------------------------------------------
        # EVALUASI SEDANG
        # ----------------------------------------------------

        elif prediksi == "SEDANG":

            evaluasi = {

                "status":
                    "Peringatan / Perlu Pengawasan",

                "badge_color":
                    "warning",

                "penjelasan":
                    "Balita menunjukkan tanda-tanda awal ketergantungan dan perlu mendapatkan pengawasan.",

                "evaluasi_ortu": [

                    "Lakukan evaluasi ulang jadwal pemakaian smartphone.",

                    "Tingkatkan peran aktif orang tua.",

                    "Gantikan durasi layar dengan permainan edukatif."
                ]
            }


        # ----------------------------------------------------
        # EVALUASI TINGGI
        # ----------------------------------------------------

        else:

            evaluasi = {

                "status":
                    "Bahaya / Risiko Tinggi",

                "badge_color":
                    "danger",

                "penjelasan":
                    "Balita terindikasi mengalami tingkat penggunaan smartphone yang tinggi.",

                "evaluasi_ortu": [

                    "Kurangi akses smartphone secara bertahap dan konsisten.",

                    "Orang tua perlu tegas dan konsisten.",

                    "Perbanyak aktivitas keluarga tanpa layar.",

                    "Jika terdapat perubahan perilaku ekstrem, pertimbangkan konsultasi dengan tenaga profesional."
                ]
            }


        # ----------------------------------------------------
        # SIMPAN RIWAYAT MYSQL
        # ----------------------------------------------------

        conn = get_db_connection()


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


            except Exception:

                pass


    return render_template(

        "diagnosa.html",

        hasil=hasil,

        probabilitas=
        probabilitas,

        evaluasi=
        evaluasi
    )


# ============================================================
# ROUTE 6
# FEATURE IMPORTANCE
# ============================================================

@app.route("/feature-importance")
def feature_importance():

    importances = (
        model.feature_importances_
    )


    fitur = [

        "Skor_X1",

        "Skor_X2",

        "Skor_X3",

        "Skor_X4"
    ]


    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )


    ax.barh(
        fitur,
        importances
    )


    ax.set_title(
        "Feature Importance Random Forest",
        fontsize=12,
        fontweight="bold"
    )


    ax.set_xlabel(
        "Nilai Importance"
    )


    ax.set_ylabel(
        "Variabel"
    )


    # --------------------------------------------------------
    # NILAI DI SAMPING BAR
    # --------------------------------------------------------

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

        img_fi=img_fi
    )


# ============================================================
# ROUTE 7
# DECISION TREE
# ============================================================

@app.route("/decision-tree")
def decision_tree():

    pohon_pertama = (
        model.estimators_[0]
    )


    fig, ax = plt.subplots(
        figsize=(14, 8)
    )


    plot_tree(

        pohon_pertama,

        feature_names=
        feature_cols,

        class_names=[
            str(c)
            for c in model.classes_
        ],

        filled=True,

        rounded=True,

        max_depth=3,

        ax=ax
    )


    ax.set_title(

        "Decision Tree #1 "
        "dalam Random Forest",

        fontsize=13,

        fontweight="bold"
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
# ROUTE 8
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


        conn = get_db_connection()


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
                    AND password = %s
                    """,

                    (
                        username,
                        password
                    )
                )


                admin_data = (
                    cursor.fetchone()
                )


                cursor.close()

                conn.close()


                if admin_data:

                    session[
                        "logged_in"
                    ] = True


                    session[
                        "admin_nama"
                    ] = admin_data[
                        "nama"
                    ]


                    return redirect(
                        url_for("admin")
                    )


            except Exception:

                pass


        # ----------------------------------------------------
        # FALLBACK LOGIN
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
            ] = "Administrator Skripsi"


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
# ROUTE 9
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


    if request.method == "POST":

        if "file_csv" in request.files:

            file = request.files[
                "file_csv"
            ]


            if (
                file
                and
                file.filename.endswith(
                    ".csv"
                )
            ):

                file.save(
                    CSV_PATH
                )


                # --------------------------------------------
                # BACA DATA BARU
                # --------------------------------------------

                df_raw, X, y = (
                    load_processed_data()
                )


                # --------------------------------------------
                # PREDIKSI
                # --------------------------------------------

                df_raw[
                    "PREDIKSI_RF"
                ] = model.predict(X)


                # --------------------------------------------
                # SIMPAN EXCEL
                # --------------------------------------------

                df_raw.to_excel(
                    EXCEL_PATH,
                    index=False
                )


                flash(

                    "Data CSV berhasil "
                    "diperbarui dan "
                    "hasil klasifikasi "
                    "berhasil disinkronkan.",

                    "success"
                )


                return redirect(
                    url_for("admin")
                )


    return render_template(
        "admin.html"
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