"""Log a finished Ultralytics run folder to Azure ML via MLflow."""
import os, sys
import mlflow
import pandas as pd

run_dir = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/runs/smoke")
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("sku110k-detector")

df = pd.read_csv(os.path.join(run_dir, "results.csv"))
df.columns = df.columns.str.strip()

def clean(name):
    # MLflow metric names can't contain brackets
    return name.replace("(B)", "").replace("(", "").replace(")", "").strip()

with mlflow.start_run(run_name=os.path.basename(run_dir)):
    mlflow.log_params({"model": "yolo11n", "epochs": 3, "imgsz": 640,
                       "batch": 8, "fraction": 0.06, "device": "cpu"})
    for _, row in df.iterrows():
        step = int(row["epoch"])
        for col in df.columns:
            if col != "epoch":
                mlflow.log_metric(clean(col), float(row[col]), step=step)
    for f in ["results.png", "confusion_matrix.png", "PR_curve.png", "args.yaml"]:
        p = os.path.join(run_dir, f)
        if os.path.exists(p):
            mlflow.log_artifact(p)
    mlflow.log_artifact(os.path.join(run_dir, "weights", "best.pt"), "weights")
print("Logged", run_dir)
