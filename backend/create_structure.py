import os

structure = [
    "app/routes/upload.py",
    "app/routes/analyze.py",
    "app/routes/preprocess.py",
    "app/routes/quality.py",
    "app/core/analysis.py",
    "app/core/preprocessing.py",
    "app/core/quality.py",
    "app/core/training.py",
    "app/common/file.py",
    "app/common/plotting.py",
    "app/models/base.py",
    "app/models/classification.py",
    "app/models/regression.py"
]

for path in structure:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()

print("Project structure created successfully!")