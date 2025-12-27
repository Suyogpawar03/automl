from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import os
import json
import uuid
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shutil
from datetime import datetime, timedelta
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
app = FastAPI()




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA

app = FastAPI()

UPLOAD_DIR = "uploads"
PLOT_DIR = "plots"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ---------------- APP SETUP ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_UPLOAD_DIR = "uploads"
os.makedirs(BASE_UPLOAD_DIR, exist_ok=True)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

DATASET_TTL_HOURS = 6

def cleanup_old_datasets():
    now = datetime.utcnow()
    cutoff = now - timedelta(hours=DATASET_TTL_HOURS)

    if not os.path.exists(BASE_UPLOAD_DIR):
        return

    for folder in os.listdir(BASE_UPLOAD_DIR):
        dataset_path = os.path.join(BASE_UPLOAD_DIR, folder)

        if not os.path.isdir(dataset_path):
            continue

        try:
            modified_time = datetime.utcfromtimestamp(
                os.path.getmtime(dataset_path)
            )

            if modified_time < cutoff:
                shutil.rmtree(dataset_path)
                print(f"🧹 Deleted old dataset: {folder}")

        except Exception as e:
            print(f"⚠️ Cleanup failed for {folder}: {e}")

def create_pdf_report(report_data: dict, output_path: str):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_path, pagesize=A4)

    elements = []

    elements.append(Paragraph("<b>AutoML Final Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"<b>Dataset ID:</b> {report_data['dataset_id']}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Generated At:</b> {report_data['generated_at']}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Analysis Summary</b>", styles["Heading2"]))
    for k, v in report_data.get("analysis", {}).items():
        elements.append(Paragraph(f"{k}: {str(v)}", styles["Normal"]))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<b>Model Performance</b>", styles["Heading2"]))

    for model, score in report_data.get("model", {}).items():
        elements.append(Paragraph(f"{model}: {score}", styles["Normal"]))

    doc.build(elements)


# ---------------- HELPERS ----------------
def dataset_dir(dataset_id: str) -> str:
    path = os.path.join(BASE_UPLOAD_DIR, dataset_id)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "plots"), exist_ok=True)
    return path


# ---------------- ROOT ----------------
@app.get("/")
def root():
    return {"status": "API is running"}


# ---------------- UPLOAD ----------------
@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    cleanup_old_datasets()  # 🧹 auto cleanup

    dataset_id = str(uuid.uuid4())
    ddir = dataset_dir(dataset_id)

    raw_path = os.path.join(ddir, "raw.csv")
    with open(raw_path, "wb") as f:
        f.write(await file.read())

    df = pd.read_csv(raw_path)

    return {
        "dataset_id": dataset_id,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist()
    }



# ---------------- ANALYZE ----------------
@app.post("/analyze")
def analyze_data(dataset_id: str):
    ddir = dataset_dir(dataset_id)
    raw_path = os.path.join(ddir, "raw.csv")

    if not os.path.exists(raw_path):
        return {"error": "Dataset not found"}

    df = pd.read_csv(raw_path)
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(exclude="number").columns.tolist()

    plots = []

    if len(numeric_columns) > 1:
        plt.figure()
        corr = df[numeric_columns].corr()
        plt.imshow(corr)
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.tight_layout()
        p = os.path.join(ddir, "plots", "correlation.png")
        plt.savefig(p)
        plt.close()
        plots.append(p)

    for col in numeric_columns[:3]:
        plt.figure()
        df[col].hist()
        p = os.path.join(ddir, "plots", f"{col}_dist.png")
        plt.savefig(p)
        plt.close()
        plots.append(p)

    analysis = {
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "statistics": df.describe().to_dict(),
        "plots": plots
    }

    with open(os.path.join(ddir, "analysis.json"), "w") as f:
        json.dump(analysis, f, indent=4)

    return analysis


# ---------------- PREPROCESS ----------------
@app.post("/preprocess")
def preprocess(dataset_id: str):
    ddir = dataset_dir(dataset_id)
    df = pd.read_csv(os.path.join(ddir, "raw.csv"))

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    if num_cols:
        df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])

    if cat_cols:
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    if num_cols:
        df[num_cols] = StandardScaler().fit_transform(df[num_cols])

    df.to_csv(os.path.join(ddir, "cleaned.csv"), index=False)
    joblib.dump(df.columns.tolist(), os.path.join(ddir, "features.pkl"))

    return {"rows": df.shape[0], "columns": df.shape[1]}


# ---------------- TRAIN ----------------

    
@app.post("/train")
def train(dataset_id: str):
    # ---------------- PATH SETUP ----------------
    dataset_path = os.path.join(UPLOAD_DIR, dataset_id)
    cleaned_path = os.path.join(dataset_path, "cleaned.csv")

    if not os.path.exists(cleaned_path):
        return {
            "error": "Run preprocess first",
            "expected_file": cleaned_path
        }

    df = pd.read_csv(cleaned_path)

    # ---------------- DATA SPLIT ----------------
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # ---------------- TARGET VALIDATION ----------------
    if y.nunique() < 2:
        return {
            "error": "Invalid target column",
            "details": "Target contains only one class."
        }

    class_counts = y.value_counts().to_dict()
    class_ratio = y.value_counts(normalize=True)
    minority_ratio = class_ratio.min()

    # 🚨 SEVERE IMBALANCE → BLOCK
    if minority_ratio < 0.01:
        return {
            "error": "Severe class imbalance detected",
            "details": {
                "class_distribution": class_counts,
                "minority_ratio": minority_ratio,
                "action": "blocked"
            }
        }

    is_classification = y.nunique() <= 10

    # ---------------- TRAIN / TEST SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if is_classification else None
    )

    results = {}
    best_model = None
    best_score = -float("inf")
    best_model_name = ""

    # ---------------- CLASSIFICATION ----------------
    if is_classification:
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "RandomForestClassifier": RandomForestClassifier(class_weight="balanced"),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "SVC": SVC(class_weight="balanced"),
            "KNN": KNeighborsClassifier()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

            results[name] = {
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
                "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
                "f1_score": f1
            }

            if f1 > best_score:
                best_score = f1
                best_model = model
                best_model_name = name

    # ---------------- REGRESSION ----------------
    else:
        models = {
            "LinearRegression": LinearRegression(),
            "RidgeRegression": Ridge(),
            "RandomForestRegressor": RandomForestRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
            "SVR": SVR()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            r2 = r2_score(y_test, preds)

            results[name] = {
                "rmse": mean_squared_error(y_test, preds, squared=False),
                "mae": mean_absolute_error(y_test, preds),
                "r2": r2
            }

            if r2 > best_score:
                best_score = r2
                best_model = model
                best_model_name = name

    # ---------------- SAVE ARTIFACTS ----------------
    os.makedirs(dataset_path, exist_ok=True)

    model_path = os.path.join(dataset_path, "model.pkl")
    metrics_path = os.path.join(dataset_path, "metrics.json")

    joblib.dump(best_model, model_path)

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    return {
        "status": "success",
        "problem_type": "classification" if is_classification else "regression",
        "best_model": best_model_name,
        "best_score": best_score,
        "class_distribution": class_counts,
        "model_path": model_path,
        "metrics_path": metrics_path,
        "all_model_metrics": results
    }




# ---------------- REPORT ----------------
@app.get("/download-report")
def download_report(dataset_id: str):
    ddir = dataset_dir(dataset_id)

    report = {
        "dataset_id": dataset_id,
        "generated_at": datetime.utcnow().isoformat(),
        "analysis": json.load(open(os.path.join(ddir, "analysis.json"))),
        "model": json.load(open(os.path.join(ddir, "model_report.json")))
    }

    path = os.path.join(ddir, "final_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=4)

    return FileResponse(path, filename="AutoML_Report.json")

@app.get("/download-report-pdf")
def download_report_pdf(dataset_id: str):
    ddir = dataset_dir(dataset_id)

    json_path = os.path.join(ddir, "final_report.json")
    if not os.path.exists(json_path):
        return {"error": "Generate JSON report first"}

    with open(json_path) as f:
        report_data = json.load(f)

    pdf_path = os.path.join(ddir, "AutoML_Report.pdf")
    create_pdf_report(report_data, pdf_path)

    return FileResponse(
        pdf_path,
        filename="AutoML_Report.pdf",
        media_type="application/pdf"
    )
@app.get("/version")
def version():
    return {"version": "dataset-folder-v2"}
