from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import json
import uuid
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, recall_score,
    f1_score, mean_squared_error, mean_absolute_error, r2_score
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

# ---------------- APP SETUP ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
PLOT_DIR = "plots"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------- ROOT ----------------
@app.get("/")
def root():
    return {"status": "API is running"}

# ---------------- UPLOAD ----------------
@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    dataset_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")

    with open(path, "wb") as f:
        f.write(await file.read())

    df = pd.read_csv(path)

    return {
        "dataset_id": dataset_id,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist()
    }

# ---------------- ANALYZE ----------------
@app.post("/analyze")
def analyze_data(dataset_id: str):
    path = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")
    if not os.path.exists(path):
        return {"error": "Dataset not found"}

    df = pd.read_csv(path)
    missing_values = df.isnull().sum().to_dict()
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(exclude="number").columns.tolist()
    stats = df.describe().to_dict()

    plots = []

    if len(numeric_columns) > 1:
        plt.figure()
        corr = df[numeric_columns].corr()
        plt.imshow(corr)
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        corr_path = os.path.join(PLOT_DIR, f"{dataset_id}_corr.png")
        plt.tight_layout()
        plt.savefig(corr_path)
        plt.close()
        plots.append(corr_path)

    for col in numeric_columns[:3]:
        plt.figure()
        df[col].hist()
        path = os.path.join(PLOT_DIR, f"{dataset_id}_{col}_dist.png")
        plt.savefig(path)
        plt.close()
        plots.append(path)

    return {
        "missing_values": missing_values,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "statistics": stats,
        "generated_plots": plots
    }

# ---------------- QUALITY CHECK ----------------
@app.post("/quality-check")
def quality_check(dataset_id: str):
    path = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")
    if not os.path.exists(path):
        return {"error": "Dataset not found"}

    df = pd.read_csv(path)
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    outliers, plots = {}, []

    for col in numeric_cols:
        series = df[col].dropna()
        median = series.median()
        mad = np.median(np.abs(series - median))
        if mad == 0:
            outliers[col] = 0
            continue

        z = 0.6745 * (series - median) / mad
        idx = series.index[np.abs(z) > 3.5].tolist()
        outliers[col] = len(idx)

        plt.figure()
        plt.scatter(df.index, df[col])
        plt.scatter(idx, df.loc[idx, col])
        p = os.path.join(PLOT_DIR, f"{dataset_id}_{col}_outliers.png")
        plt.savefig(p)
        plt.close()
        plots.append(p)

    target = numeric_cols[-1] if len(numeric_cols) >= 2 else None
    dependency = {
        col: df[col].corr(df[target])
        for col in numeric_cols if col != target
    } if target else {}

    weak_features = [k for k, v in dependency.items() if abs(v) < 0.1]

    return {
        "outliers": outliers,
        "target": target,
        "dependency": dependency,
        "weak_features": weak_features,
        "generated_plots": plots
    }

# ---------------- PREPROCESS ----------------
@app.post("/preprocess")
def preprocess(dataset_id: str):
    path = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")
    if not os.path.exists(path):
        return {"error": "Dataset not found"}

    df = pd.read_csv(path)

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    if num_cols:
        df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])

    if cat_cols:
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    clean_path = os.path.join(UPLOAD_DIR, f"{dataset_id}_cleaned.csv")
    df.to_csv(clean_path, index=False)

    joblib.dump(df.columns.tolist(), os.path.join(UPLOAD_DIR, f"{dataset_id}_features.pkl"))

    return {"rows": df.shape[0], "columns": df.shape[1]}

# ---------------- TRAIN ----------------
@app.post("/train")
def train(dataset_id: str):
    path = os.path.join(UPLOAD_DIR, f"{dataset_id}_cleaned.csv")
    if not os.path.exists(path):
        return {"error": "Run preprocess first"}

    df = pd.read_csv(path)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    is_classification = y.nunique() <= 10

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)

    models = (
        {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForestClassifier": RandomForestClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "SVC": SVC(),
            "KNN": KNeighborsClassifier()
        }
        if is_classification else
        {
            "LinearRegression": LinearRegression(),
            "RidgeRegression": Ridge(),
            "RandomForestRegressor": RandomForestRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
            "SVR": SVR()
        }
    )

    results, best_model, best_score = {}, None, -1

    for name, model in models.items():
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        score = f1_score(yte, preds, average="weighted") if is_classification else r2_score(yte, preds)
        results[name] = score
        if score > best_score:
            best_score, best_model = score, model

    model_path = os.path.join(UPLOAD_DIR, f"{dataset_id}_best_model.pkl")
    joblib.dump(best_model, model_path)

    return {
        "problem_type": "classification" if is_classification else "regression",
        "best_score": best_score,
        "model_path": model_path,
        "all_model_scores": results
    }

# ---------------- UNSUPERVISED ----------------
@app.post("/unsupervised")
def unsupervised(dataset_id: str):
    path = os.path.join(UPLOAD_DIR, f"{dataset_id}_cleaned.csv")
    if not os.path.exists(path):
        return {"error": "Run preprocess first"}

    df = pd.read_csv(path)
    X = df.select_dtypes(include=["int64", "float64"])

    kmeans = KMeans(n_clusters=3).fit_predict(X)
    dbscan = DBSCAN().fit_predict(X)
    agg = AgglomerativeClustering(n_clusters=3).fit_predict(X)

    pca = PCA(2).fit_transform(X)
    plt.figure()
    plt.scatter(pca[:, 0], pca[:, 1])
    p = os.path.join(PLOT_DIR, f"{dataset_id}_pca.png")
    plt.savefig(p)
    plt.close()

    return {
        "kmeans": kmeans.tolist(),
        "dbscan": dbscan.tolist(),
        "agglomerative": agg.tolist(),
        "pca_plot": p
    }

# ---------------- PREDICT ----------------
@app.post("/predict")
def predict(dataset_id: str, input_data: list[dict]):
    model_path = os.path.join(UPLOAD_DIR, f"{dataset_id}_best_model.pkl")
    cols_path = os.path.join(UPLOAD_DIR, f"{dataset_id}_features.pkl")

    if not os.path.exists(model_path):
        return {"error": "Train model first"}

    model = joblib.load(model_path)
    cols = joblib.load(cols_path)

    df = pd.DataFrame(input_data)
    for col in cols:
        if col not in df:
            df[col] = 0

    df = df[cols]
    preds = model.predict(df)

    return {"predictions": preds.tolist()}



