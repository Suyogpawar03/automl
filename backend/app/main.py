from dotenv import load_dotenv
import os
import json

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


from openai import OpenAI
client_ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))







from fastapi import FastAPI, UploadFile, File
import pandas as pd
import os
import json
import joblib
import requests
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge
)
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

latest_file_path = None


@app.get("/")
def root():
    return {"status": "API is running"}


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    global latest_file_path

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    latest_file_path = file_path
    df = pd.read_csv(file_path)

    return {
        "message": "File uploaded successfully",
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist()
    }


@app.get("/analyze")
def analyze_data():
    if latest_file_path is None:
        return {"error": "No dataset uploaded yet"}

    df = pd.read_csv(latest_file_path)

    # ---------- BASIC ANALYSIS ----------
    missing_values = df.isnull().sum().to_dict()
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(exclude="number").columns.tolist()

    stats = df.describe().to_dict()

    # ---------- VISUALIZATION ----------
    plot_files = []

    # Correlation Matrix
    if len(numeric_columns) > 1:
        corr = df[numeric_columns].corr()
        plt.figure()
        plt.imshow(corr)
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Correlation Matrix")

        corr_path = os.path.join(PLOT_DIR, "correlation_matrix.png")
        plt.tight_layout()
        plt.savefig(corr_path)
        plt.close()

        plot_files.append(corr_path)

    # Distribution plots
    for col in numeric_columns[:3]:
        plt.figure()
        df[col].hist()
        plt.title(f"Distribution of {col}")

        path = os.path.join(PLOT_DIR, f"{col}_distribution.png")
        plt.savefig(path)
        plt.close()

        plot_files.append(path)
 
    analysis_report = {
    "missing_values": missing_values,
    "numeric_columns": numeric_columns,
    "categorical_columns": categorical_columns,
    "statistics": stats
}

    with open(os.path.join(UPLOAD_DIR, "analysis_report.json"), "w") as f:
     json.dump(analysis_report, f, indent=4)

    return {
        "missing_values": missing_values,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "statistics": stats,
        "generated_plots": plot_files
    }


@app.get("/quality-check")
def quality_check():
    import numpy as np

    files = os.listdir(UPLOAD_DIR)
    if not files:
        return {"error": "No dataset uploaded"}

    latest_file = max(
        [os.path.join(UPLOAD_DIR, f) for f in files],
        key=os.path.getctime
    )

    df = pd.read_csv(latest_file)

    report = {}
    plots = []

    # Detect numeric columns safely
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # ------------------ OUTLIER DETECTION (MAD) ------------------
    outliers = {}

    for col in numeric_cols:
        series = df[col].dropna()

        if series.empty:
            outliers[col] = 0
            continue

        median = series.median()
        mad = np.median(np.abs(series - median))

        # Avoid division by zero
        if mad == 0:
            outliers[col] = 0
            continue

        modified_z = 0.6745 * (series - median) / mad
        outlier_indices = series.index[np.abs(modified_z) > 3.5].tolist()

        outliers[col] = len(outlier_indices)

        # Scatter plot for visibility
        plt.figure()
        plt.scatter(df.index, df[col], alpha=0.7)
        plt.scatter(
            outlier_indices,
            df.loc[outlier_indices, col],
            alpha=1.0
        )
        plt.title(f"Modified Z-Score Outliers in {col}")
        plt.xlabel("Row Index")
        plt.ylabel(col)

        path = os.path.join(PLOT_DIR, f"{col}_mad_outliers.png")
        plt.savefig(path)
        plt.close()
        plots.append(path)

    report["outliers"] = outliers

    # ------------------ FEATURE DEPENDENCY ------------------
    dependency = {}
    target = None

    if len(numeric_cols) >= 2:
        target = numeric_cols[-1]  # auto-select last numeric column as target

        for col in numeric_cols:
            if col == target:
                continue
            corr = df[col].corr(df[target])
            dependency[col] = corr

    weak_features = [
        col for col, corr in dependency.items()
        if corr is not None and abs(corr) < 0.1
    ]

    report["target"] = target
    report["feature_dependency"] = dependency
    report["weak_features"] = weak_features

    return {
        "dataset_used": os.path.basename(latest_file),
        "numeric_columns_detected": numeric_cols,
        "report": report,
        "generated_plots": plots
    }


from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

@app.post("/preprocess")
def preprocess():
    try:
        global latest_file_path

        if not latest_file_path or not os.path.exists(latest_file_path):
            return {"error": "No dataset uploaded. Run /upload first."}

        df = pd.read_csv(latest_file_path)

        if df.empty:
            return {"error": "Uploaded dataset is empty."}

        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

        # -------- Missing Values --------
        if num_cols:
            df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])

        if cat_cols:
            df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

        # -------- Outlier Capping --------
        for col in num_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

        # -------- Encoding --------
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # -------- Scaling --------
        if num_cols:
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])

        # -------- Save artifacts --------
        cleaned_path = os.path.join(UPLOAD_DIR, "cleaned_dataset.csv")
        df.to_csv(cleaned_path, index=False)

        joblib.dump(list(df.columns), os.path.join(UPLOAD_DIR, "feature_columns.pkl"))

        return {
            "message": "Preprocessing completed successfully",
            "rows": df.shape[0],
            "columns": df.shape[1]
        }

    except Exception as e:
        return {
            "error": "Preprocessing failed",
            "details": str(e)
        }


@app.post("/train")
def train():
    path = os.path.join(UPLOAD_DIR, "cleaned_dataset.csv")
    if not os.path.exists(path):
        return {"error": "Run preprocessing first"}

    df = pd.read_csv(path)

    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Auto-detect problem type
    is_classification = y.nunique() <= 10

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}
    best_model = None
    best_score = -1
    best_model_name = ""

    # ---------------- CLASSIFICATION ----------------
    if is_classification:
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForestClassifier": RandomForestClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "SVC": SVC(),
            "KNN": KNeighborsClassifier()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            score = f1_score(y_test, preds, average="weighted")

            results[name] = {
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, average="weighted"),
                "recall": recall_score(y_test, preds, average="weighted"),
                "f1_score": score
            }

            if score > best_score:
                best_score = score
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

            score = r2_score(y_test, preds)

            results[name] = {
                "rmse": mean_squared_error(y_test, preds, squared=False),
                "mae": mean_absolute_error(y_test, preds),
                "r2": score
            }

            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = name

    # Save best model and metrics
    model_path = os.path.join(UPLOAD_DIR, "best_model.pkl")
    metrics_path = os.path.join(UPLOAD_DIR, "metrics.json")

    joblib.dump(best_model, model_path)
    with open(os.path.join(UPLOAD_DIR, "model_report.json"), "w") as f:
     json.dump(results, f, indent=4)


    return {
        "problem_type": "classification" if is_classification else "regression",
        "best_model": best_model_name,
        "best_score": best_score,
        "model_path": model_path,
        "metrics_path": metrics_path,
        "all_model_metrics": results
    }

 #-----------unsupervised------------


@app.post("/unsupervised")
def unsupervised():
    path = os.path.join(UPLOAD_DIR, "cleaned_dataset.csv")
    if not os.path.exists(path):
        return {"error": "Run preprocessing first"}

    df = pd.read_csv(path)

    # Use only numeric features
    X = df.select_dtypes(include=["int64", "float64"])

    if X.empty:
        return {"error": "No numeric columns for unsupervised learning"}

    results = {}

    # ---------------- KMEANS ----------------
    kmeans = KMeans(n_clusters=3, random_state=42)
    results["KMeans"] = kmeans.fit_predict(X).tolist()

    # ---------------- DBSCAN ----------------
    dbscan = DBSCAN()
    results["DBSCAN"] = dbscan.fit_predict(X).tolist()

    # ---------------- AGGLOMERATIVE ----------------
    agg = AgglomerativeClustering(n_clusters=3)
    results["Agglomerative"] = agg.fit_predict(X).tolist()

    # ---------------- PCA VISUALIZATION ----------------
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)

    plt.figure()
    plt.scatter(reduced[:, 0], reduced[:, 1])
    plt.title("PCA Visualization (Unsupervised)")
    pca_path = os.path.join(PLOT_DIR, "unsupervised_pca.png")
    plt.savefig(pca_path)
    plt.close()

    return {
        "models_run": ["KMeans", "DBSCAN", "Agglomerative"],
        "cluster_assignments": results,
        "pca_plot": pca_path
    }

@app.post("/predict")
def predict(input_data: list[dict]):
    try:
        model_path = os.path.join(UPLOAD_DIR, "best_model.pkl")
        cols_path = os.path.join(UPLOAD_DIR, "feature_columns.pkl")

        if not os.path.exists(model_path):
            return {"error": "best_model.pkl not found. Train model first."}

        if not os.path.exists(cols_path):
            return {"error": "feature_columns.pkl not found. Run preprocess again."}

        model = joblib.load(model_path)
        expected_cols = joblib.load(cols_path)

        df_input = pd.DataFrame(input_data)

        if df_input.empty:
            return {"error": "Input data is empty"}

        # Convert values to numeric
        for col in df_input.columns:
            df_input[col] = pd.to_numeric(df_input[col], errors="coerce")

        # Add missing columns
        for col in expected_cols:
            if col not in df_input.columns:
                df_input[col] = 0

        # Remove extra columns
        df_input = df_input[expected_cols]

        predictions = model.predict(df_input)

        return {
            "status": "success",
            "predictions": predictions.tolist(),
            "rows": len(df_input)
        }

    except Exception as e:
        return {
            "error": "Prediction failed",
            "details": str(e)
        }


