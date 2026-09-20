import os
import io
import json
import uuid
import base64
import warnings

warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree


app = Flask(__name__)
app.secret_key = "skripsi_kecanduan_balita_secret_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ORIGINAL_DATASET = os.path.join(BASE_DIR, "tabulasi_data_terbaru(1)(1).xlsx")
ORIGINAL_MODEL = os.path.join(BASE_DIR, "model_random_forest_110_40.pkl")
ORIGINAL_EXCEL = os.path.join(BASE_DIR, "Hasil_RF_110_40.xlsx")

DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")
EXCEL_DIR = os.path.join(BASE_DIR, "hasil_excel")
REGISTRY_PATH = os.path.join(BASE_DIR, "dataset_registry.json")

for p in [DATASET_DIR, MODEL_DIR, EXCEL_DIR]:
    os.makedirs(p, exist_ok=True)

FEATURE_COLS = ["Skor_X1", "Skor_X2", "Skor_X3", "Skor_X4"]
TARGET = "LABEL"
LOCATION = "Jarak_Kota"

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"


def admin_required():
    return bool(session.get("logged_in", False))


def load_registry():
    if not os.path.exists(REGISTRY_PATH):
        return []
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_registry(data):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def fig_to_base64(fig):
    bio = io.BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight", dpi=120)
    bio.seek(0)
    return base64.b64encode(bio.getvalue()).decode("utf-8")


def load_model(path):
    if not os.path.exists(path):
        return None
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


def read_excel_dataset(path):
    df = pd.read_excel(path, sheet_name="TABULASI", header=[0, 1])
    cols = []
    for h1, h2 in df.columns:
        h1, h2 = str(h1).strip(), str(h2).strip()
        if h1 == "RESP":
            name = "No_Responden"
        elif h1 == "Tempat Tinggal":
            name = "Jarak_Kota"
        elif h1 == "SKOR TOTAL Y":
            name = "SKOR_TOTAL_Y"
        elif h1 == "LABEL":
            name = "LABEL"
        else:
            name = h2
        cols.append(name)
    df.columns = cols
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def read_csv_dataset(path):
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Unnamed: 0": "No_Responden",
        "Tempat Tinggal": "Jarak_Kota",
        "TOTAL X1": "Skor_X1",
        "TOTAL X2": "Skor_X2",
        "TOTAL X3": "Skor_X3",
        "TOTAL X4": "Skor_X4",
        "Unnamed: 21": "SKOR_TOTAL_Y",
        "Unnamed: 22": "LABEL",
        "Unnamed: 23": "LABEL",
    })
    return df


def prepare_dataset(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Lokasi
    if LOCATION not in df.columns:
        for alt in ["Tempat Tinggal", "Tempat_Tinggal", "Jarak Kota", "Lokasi"]:
            if alt in df.columns:
                df[LOCATION] = df[alt]
                break

    if LOCATION not in df.columns:
        df[LOCATION] = "Tidak Diketahui"

    df[LOCATION] = df[LOCATION].astype(str).str.strip().str.upper()
    df[LOCATION] = df[LOCATION].replace({
        "JAUH": "JAUH DARI KOTA",
        "DEKAT": "DEKAT KOTA",
    })

    # Label
    if TARGET not in df.columns:
        raise ValueError("Kolom LABEL tidak ditemukan pada dataset.")
    df[TARGET] = df[TARGET].astype(str).str.strip().str.upper()

    # Skor X1-X4
    groups = {
        "Skor_X1": ["X1.1", "X1.2", "X1.3"],
        "Skor_X2": ["X2.1", "X2.2", "X2.3"],
        "Skor_X3": ["X3.1", "X3.2", "X3.3"],
        "Skor_X4": ["X4.1", "X4.2", "X4.3"],
    }
    for score, cols in groups.items():
        if all(c in df.columns for c in cols):
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df[score] = df[cols].sum(axis=1)
        elif score in df.columns:
            df[score] = pd.to_numeric(df[score], errors="coerce")

    # Total Y
    ycols = ["Y1", "Y2", "Y3", "Y4"]
    if all(c in df.columns for c in ycols):
        for c in ycols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["Total_Y"] = df[ycols].sum(axis=1)
    elif "SKOR_TOTAL_Y" in df.columns:
        df["Total_Y"] = pd.to_numeric(df["SKOR_TOTAL_Y"], errors="coerce")

    for c in FEATURE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    missing = [c for c in FEATURE_COLS + [TARGET, LOCATION] if c not in df.columns]
    if missing:
        raise ValueError("Kolom berikut tidak ditemukan: " + ", ".join(missing))

    df = df.dropna(subset=FEATURE_COLS + [TARGET, LOCATION]).reset_index(drop=True)

    if "No_Responden" not in df.columns:
        df.insert(0, "No_Responden", range(1, len(df) + 1))

    return df


def load_dataset_file(path):
    if not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = read_excel_dataset(path)
    elif ext == ".csv":
        df = read_csv_dataset(path)
    else:
        raise ValueError("Format file tidak didukung.")
    return prepare_dataset(df)


def split_110_40(df):
    # Penelitian utama: 88 jauh -> 65 train + 23 test;
    # 62 dekat -> 45 train + 17 test.
    if len(df) == 150:
        jauh = df.index[df[LOCATION] == "JAUH DARI KOTA"].tolist()
        dekat = df.index[df[LOCATION] == "DEKAT KOTA"].tolist()
        if len(jauh) == 88 and len(dekat) == 62:
            rng = np.random.RandomState(42)
            rng.shuffle(jauh)
            rng.shuffle(dekat)
            train_idx = sorted(jauh[:65] + dekat[:45])
            test_idx = sorted(jauh[65:] + dekat[45:])
            return df.loc[train_idx].copy(), df.loc[test_idx].copy()

    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df[TARGET]
    )
    return train_df.copy(), test_df.copy()


def evaluate_model(model, train_df, test_df):
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET]
    y_pred = model.predict(X_test)

    labels = list(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    hasil_uji = test_df.copy()
    hasil_uji["Prediksi_RF"] = y_pred

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "cm": cm,
        "report": report,
        "labels": labels,
        "hasil_uji": hasil_uji,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def get_active_dataset():
    dataset_id = session.get("active_dataset", "original")

    if dataset_id == "original":
        df = load_dataset_file(ORIGINAL_DATASET)
        model = load_model(ORIGINAL_MODEL)
        if df is None:
            raise FileNotFoundError(
                "Dataset tidak ditemukan: " + ORIGINAL_DATASET
            )
        if model is None:
            raise FileNotFoundError(
                "Model tidak ditemukan: " + ORIGINAL_MODEL
            )
        return {
            "id": "original",
            "name": "Data Penelitian Terbaru",
            "file": os.path.basename(ORIGINAL_DATASET),
            "model_file": os.path.basename(ORIGINAL_MODEL),
            "excel_file": os.path.basename(ORIGINAL_EXCEL),
            "path": ORIGINAL_DATASET,
            "model_path": ORIGINAL_MODEL,
            "excel_path": ORIGINAL_EXCEL,
            "df": df,
            "model": model,
        }

    for item in load_registry():
        if item.get("id") == dataset_id:
            dataset_path = os.path.join(DATASET_DIR, item["dataset_file"])
            model_path = os.path.join(MODEL_DIR, item["model_file"])
            excel_path = os.path.join(EXCEL_DIR, item["excel_file"])
            df = load_dataset_file(dataset_path)
            model = load_model(model_path)
            if df is not None and model is not None:
                return {
                    "id": item["id"],
                    "name": item["name"],
                    "file": item["dataset_file"],
                    "model_file": item["model_file"],
                    "excel_file": item["excel_file"],
                    "path": dataset_path,
                    "model_path": model_path,
                    "excel_path": excel_path,
                    "df": df,
                    "model": model,
                }

    session["active_dataset"] = "original"
    return get_active_dataset()


@app.route("/")
@app.route("/dashboard")
def dashboard():
    dataset = get_active_dataset()
    df, model = dataset["df"], dataset["model"]
    train_df, test_df = split_110_40(df)
    result = evaluate_model(model, train_df, test_df)

    labels = result["labels"]
    cm = result["cm"]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion Matrix")
    img_cm = fig_to_base64(fig)
    plt.close(fig)

    actual = [int(np.sum(result["y_test"] == x)) for x in labels]
    pred = [int(np.sum(result["y_pred"] == x)) for x in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - .175, actual, .35, label="Aktual")
    ax.bar(x + .175, pred, .35, label="Prediksi")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Jumlah Responden")
    ax.set_title("Aktual vs Prediksi")
    ax.legend()
    img_bar = fig_to_base64(fig)
    plt.close(fig)

    far_total = int((df[LOCATION] == "JAUH DARI KOTA").sum())
    near_total = int((df[LOCATION] == "DEKAT KOTA").sum())
    far_high = int(((df[LOCATION] == "JAUH DARI KOTA") & (df[TARGET] == "TINGGI")).sum())
    near_high = int(((df[LOCATION] == "DEKAT KOTA") & (df[TARGET] == "TINGGI")).sum())
    far_pct = far_high / far_total * 100 if far_total else 0
    near_pct = near_high / near_total * 100 if near_total else 0

    loc_test = test_df[LOCATION].value_counts().reindex(
        ["JAUH DARI KOTA", "DEKAT KOTA"], fill_value=0
    ).rename_axis("Jarak Kota").reset_index(name="Jumlah")

    report_df = pd.DataFrame(result["report"]).transpose().round(3)

    return render_template(
        "dashboard.html",
        accuracy=result["accuracy"] * 100,
        total_data=len(df),
        training_data=len(train_df),
        testing_data=len(test_df),
        jumlah_tree=int(getattr(model, "n_estimators", 100)),
        jumlah_fitur=len(FEATURE_COLS),
        model_name=dataset["name"],
        near_total=near_total,
        far_total=far_total,
        near_high=near_high,
        far_high=far_high,
        near_pct=near_pct,
        far_pct=far_pct,
        img_cm=img_cm,
        img_bar=img_bar,
        location_test=loc_test.to_html(classes="data-table", index=False, border=0),
        report_table=report_df.to_html(classes="data-table", border=0),
    )


@app.route("/tabulasi")
def tabulasi():
    df = get_active_dataset()["df"]
    return render_template(
        "tabulasi.html",
        table_html=df.to_html(classes="data-table", index=False, border=0),
        total_data=len(df),
    )


@app.route("/skor-target")
def skor_target():
    dataset = get_active_dataset()
    df, model = dataset["df"], dataset["model"]

    cols = [
        "No_Responden", "Jarak_Kota",
        "X1.1", "X1.2", "X1.3", "Skor_X1",
        "X2.1", "X2.2", "X2.3", "Skor_X2",
        "X3.1", "X3.2", "X3.3", "Skor_X3",
        "X4.1", "X4.2", "X4.3", "Skor_X4",
        "Total_Y", "LABEL"
    ]
    cols = [c for c in cols if c in df.columns]

    score_cols = ["No_Responden", "Jarak_Kota", "Skor_X1", "Skor_X2", "Skor_X3", "Skor_X4", "Total_Y", "LABEL"]
    score_cols = [c for c in score_cols if c in df.columns]

    return render_template(
        "skor_target.html",
        detail_table=df[cols].to_html(classes="data-table", index=False, border=0),
        skor_table=df[score_cols].to_html(classes="data-table", index=False, border=0),
        target_table=df[TARGET].value_counts().rename_axis("Kategori").reset_index(name="Jumlah").to_html(classes="data-table", index=False, border=0),
        model_info={
            "nama_file": dataset["model_file"],
            "tipe_model": type(model).__name__,
            "jumlah_pohon": getattr(model, "n_estimators", 100),
            "random_state": getattr(model, "random_state", 42),
        },
    )


@app.route("/excel")
@app.route("/excel-view")
def excel_view():
    path = get_active_dataset()["excel_path"]
    parts = []
    if os.path.exists(path):
        try:
            xls = pd.ExcelFile(path)
            for sheet in xls.sheet_names:
                d = pd.read_excel(path, sheet_name=sheet)
                parts.append(
                    f'<div class="excel-sheet"><h3>{sheet}</h3>'
                    + d.to_html(classes="data-table", index=False, border=0)
                    + "</div>"
                )
        except Exception as e:
            parts.append(f'<div class="alert danger">Gagal membaca Excel: {e}</div>')
    else:
        parts.append(
            '<div class="empty-state">File hasil Excel belum tersedia. '
            'Pastikan <b>Hasil_RF_110_40.xlsx</b> ada di folder project.</div>'
        )
    return render_template(
        "excel_view.html",
        excel_table="".join(parts),
        excel_file=os.path.basename(path),
    )


@app.route("/analisis-lokasi")
def analisis_lokasi():
    dataset = get_active_dataset()
    df, model = dataset["df"], dataset["model"]
    train_df, test_df = split_110_40(df)
    result = evaluate_model(model, train_df, test_df)
    hu = result["hasil_uji"]

    total_lokasi = df[LOCATION].value_counts().to_dict()
    train_lokasi = train_df[LOCATION].value_counts().to_dict()
    test_lokasi = test_df[LOCATION].value_counts().to_dict()

    tabel_label = hu.groupby(LOCATION)[TARGET].value_counts().unstack(fill_value=0)
    tabel_prediksi = hu.groupby(LOCATION)["Prediksi_RF"].value_counts().unstack(fill_value=0)

    tinggi_aktual = hu[hu[TARGET] == "TINGGI"].groupby(LOCATION).size().to_dict()
    tinggi_prediksi = hu[hu["Prediksi_RF"] == "TINGGI"].groupby(LOCATION).size().to_dict()

    return render_template(
        "analisis_lokasi.html",
        total_lokasi=total_lokasi,
        train_lokasi=train_lokasi,
        test_lokasi=test_lokasi,
        tabel_label=tabel_label.to_html(classes="data-table", border=0),
        tabel_prediksi=tabel_prediksi.to_html(classes="data-table", border=0),
        kecanduan=tinggi_aktual,
        tinggi_prediksi=tinggi_prediksi,
        hasil_uji=hu.to_dict(orient="records"),
    )


@app.route("/diagnosa", methods=["GET", "POST"])
def diagnosa():
    hasil = None
    probabilitas = None
    model = get_active_dataset()["model"]

    if request.method == "POST":
        try:
            vals = [float(request.form[f"x{i}"]) for i in range(1, 5)]
            inp = pd.DataFrame([vals], columns=FEATURE_COLS)
            hasil = model.predict(inp)[0]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(inp)[0]
                probabilitas = {
                    str(c): f"{p*100:.1f}%"
                    for c, p in zip(model.classes_, probs)
                }
        except Exception as e:
            flash(f"Input tidak valid: {e}", "danger")

    return render_template("diagnosa.html", hasil=hasil, probabilitas=probabilitas)


@app.route("/feature-importance")
def feature_importance():
    model = get_active_dataset()["model"]
    vals = model.feature_importances_

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(FEATURE_COLS, vals)
    ax.set_title("Feature Importance Random Forest")
    ax.set_xlabel("Nilai Importance")
    for i, v in enumerate(vals):
        ax.text(v + .01, i, f"{v:.3f}", va="center")
    plt.tight_layout()
    img = fig_to_base64(fig)
    plt.close(fig)
    return render_template("feature_importance.html", img_fi=img)


@app.route("/decision-tree")
def decision_tree():
    model = get_active_dataset()["model"]
    total_trees = int(getattr(model, "n_estimators", len(getattr(model, "estimators_", [])) or 100))

    try:
        tree_number = int(request.args.get("tree", 1))
    except ValueError:
        tree_number = 1
    tree_number = max(1, min(tree_number, total_trees))

    tree = model.estimators_[tree_number - 1]
    fig, ax = plt.subplots(figsize=(18, 10))
    plot_tree(
        tree,
        feature_names=FEATURE_COLS,
        class_names=[str(c) for c in model.classes_],
        filled=True,
        rounded=True,
        max_depth=3,
        ax=ax,
        impurity=True,
    )
    ax.set_title(f"Decision Tree #{tree_number} dalam Random Forest")
    img_tree = fig_to_base64(fig)
    plt.close(fig)

    return render_template(
        "decision_tree.html",
        img_tree=img_tree,
        tree_number=tree_number,
        total_trees=total_trees,
        previous_tree=max(1, tree_number - 1),
        next_tree=min(total_trees, tree_number + 1),
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["logged_in"] = True
            session["admin_nama"] = "Administrator"
            flash("Login admin berhasil.", "success")
            return redirect(url_for("admin"))
        flash("Username atau password salah.", "danger")
    return render_template("login.html")


@app.route("/admin", methods=["GET", "POST"])
def admin():
    if not admin_required():
        return redirect(url_for("login"))

    # Kompatibel dengan form admin lama maupun baru.
    if request.method == "POST":
        action = request.form.get("action", "")

        if action in ["select_dataset", "select"]:
            dataset_id = request.form.get("dataset_id", "original")
            valid = dataset_id == "original" or any(x.get("id") == dataset_id for x in load_registry())
            if valid:
                session["active_dataset"] = dataset_id
                flash("Dataset aktif berhasil diubah.", "success")
            else:
                flash("Dataset tidak ditemukan.", "danger")
            return redirect(url_for("admin"))

        if action in ["upload_dataset", "upload"]:
            return _process_upload(request)

        # Form lama: POST /admin dengan file_csv
        if "file_csv" in request.files:
            return _process_upload(request, field_name="file_csv")

    dataset = get_active_dataset()
    df, model = dataset["df"], dataset["model"]
    train_df, test_df = split_110_40(df)
    result = evaluate_model(model, train_df, test_df)

    dataset_list = [{
        "id": "original",
        "name": "Data Penelitian Terbaru",
        "file": os.path.basename(ORIGINAL_DATASET),
        "accuracy": round(result["accuracy"] * 100, 2),
        "total_data": len(df),
    }]

    for item in load_registry():
        dataset_list.append({
            "id": item["id"],
            "name": item["name"],
            "file": item["dataset_file"],
            "accuracy": item["accuracy"],
            "total_data": item["total_data"],
        })

    return render_template(
        "admin.html",
        model_name=dataset["name"],
        dataset_file=dataset["file"],
        total_data=len(df),
        training_data=len(train_df),
        testing_data=len(test_df),
        jumlah_tree=getattr(model, "n_estimators", 100),
        accuracy=result["accuracy"] * 100,
        dataset_list=dataset_list,
        active_dataset=dataset["id"],
    )


@app.route("/admin/upload", methods=["POST"])
def upload_dataset():
    if not admin_required():
        return redirect(url_for("login"))
    return _process_upload(request, field_name="file_dataset")


def _process_upload(req, field_name="file_dataset"):
    file = req.files.get(field_name)
    if not file or not file.filename:
        flash("File belum dipilih.", "danger")
        return redirect(url_for("admin"))

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".xlsx", ".xls", ".csv"]:
        flash("File harus Excel (.xlsx/.xls) atau CSV.", "danger")
        return redirect(url_for("admin"))

    try:
        dataset_id = "dataset_" + uuid.uuid4().hex[:8]
        original_name = secure_filename(file.filename)
        dataset_filename = dataset_id + "_" + original_name
        dataset_path = os.path.join(DATASET_DIR, dataset_filename)
        file.save(dataset_path)

        df = load_dataset_file(dataset_path)
        train_df, test_df = split_110_40(df)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(train_df[FEATURE_COLS], train_df[TARGET])
        result = evaluate_model(model, train_df, test_df)

        model_filename = dataset_id + "_model.pkl"
        model_path = os.path.join(MODEL_DIR, model_filename)
        joblib.dump(model, model_path)

        excel_filename = dataset_id + "_hasil.xlsx"
        excel_path = os.path.join(EXCEL_DIR, excel_filename)

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Data", index=False)
            train_df.to_excel(writer, sheet_name="Data Latih", index=False)
            result["hasil_uji"].to_excel(writer, sheet_name="Data Uji", index=False)
            pd.DataFrame(
                result["cm"], index=result["labels"], columns=result["labels"]
            ).to_excel(writer, sheet_name="Confusion Matrix")
            pd.DataFrame(result["report"]).transpose().to_excel(
                writer, sheet_name="Classification Report"
            )

        registry = load_registry()
        registry.append({
            "id": dataset_id,
            "name": original_name,
            "dataset_file": dataset_filename,
            "model_file": model_filename,
            "excel_file": excel_filename,
            "total_data": len(df),
            "training_data": len(train_df),
            "testing_data": len(test_df),
            "accuracy": round(result["accuracy"] * 100, 2),
            "jumlah_tree": 100,
        })
        save_registry(registry)
        session["active_dataset"] = dataset_id

        flash(
            f"Dataset berhasil diproses: {len(train_df)} training + {len(test_df)} testing. "
            f"Akurasi {result['accuracy']*100:.2f}%.",
            "success",
        )
    except Exception as e:
        flash(f"Gagal memproses dataset: {e}", "danger")

    return redirect(url_for("admin"))


@app.route("/logout")
def logout():
    session.clear()
    flash("Logout berhasil.", "success")
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    app.run(debug=True)
