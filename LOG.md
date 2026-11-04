# Project log

## Day 1 (Wed 4 Nov 2026): setup and smoke test
- Azure: resource group rg-retail, workspace mlw-retail, storage stretail26561 (Australia East), datastore retaildata.
- GPU quota: none usable on the student subscription (only retired NC/NV families). Training moves to Kaggle GPUs; Azure handles data, tracking and deployment.
- Instacart: original Kaggle competition page is gone; used mirror yasserh/instacart-online-grocery-basket-analysis-dataset, downloaded via browser (the Kaggle API failed with the new token format). Uploaded to raw/instacart/, registered as data asset instacart-raw v1.
- SKU-110K: downloaded and converted via the Ultralytics SKU-110K.yaml. A few images give harmless "Corrupt JPEG data" warnings.
- Smoke test: YOLO11n, about 500 train images (fraction=0.06), 3 epochs, CPU, about 21 min.
  Val (588 images): P 0.735, R 0.623, mAP50 0.674, mAP50-95 0.345.
- MLflow fixes: clicked JupyterLab to refresh the compute instance sign-in; pinned mlflow<3 for artifact uploads; logged with src/log_run_to_azure.py to experiment sku110k-detector.
- Lessons: PowerShell vs bash syntax; keep data on local disk, not cloudfiles; git safe.directory needed on cloudfiles.