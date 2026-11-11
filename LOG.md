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

## Day 2 (Thu 5 Nov 2026): full training and EDA

### Training (Kaggle GPU)
- Notebook: kaggle_train_sku110k.ipynb. YOLO11s, imgsz 640, batch 16, 4-hour cap, patience 15, max_det 1000.
- First run crashed: dataset downloaded into /kaggle/working (20 GB output cap) and filled the disk.
  Fix: copied SKU-110K.yaml with an absolute path to /tmp/datasets.
- Many "Corrupt JPEG data" warnings: harmless, a few images have damaged file endings.
- Results: epochs completed ___, best val mAP50 ___, test P ___ / R ___ / mAP50 ___ / mAP50-95 ___
  (smoke test baseline: P 0.735, R 0.623, mAP50 0.674, mAP50-95 0.345).

### Instacart EDA
- 3.2M prior orders, 32.4M order lines, 206K customers, 49.7K products.
- Pattern 1, baskets: median 8 products (mean 10.1, max 145); customers place a median of 10 orders (4 to 100).
- Pattern 2, loyalty: 59% of items are reorders. Dairy/eggs (~67%), beverages, produce and bakery are habitual;
  personal care (~32%), pantry and international are mostly one-off. 19 of the top 20 products are produce;
  bananas appear in ~15% of orders.
- Pattern 3, timing: orders cluster 9am to 4pm; days 0 and 1 busiest (likely the weekend).
  Gaps between orders peak at 7 days (weekly shoppers), with bumps at 14 and 21; 30 is a capped value.
- Data quirk: a department named "missing"; treat it as unknown.
- Saved as Parquet (713 MB CSV to 149 MB), uploaded to processed/instacart/, data asset instacart-processed v4.

### SKU-110K EDA
- Median ~138 products per image (val max 718); only 1.1% of images exceed 300, so max_det isn't the main recall issue.
- Products are ~5% of image width; photos are ~2448x3264, so at imgsz 640 a product is only ~30 px wide.
  Day 3 experiment: train at a larger image size (e.g. 1024) to improve recall.

### Fixes
- AzCopy sign-in is lost when the compute instance restarts: set AZCOPY_AUTO_LOGIN_TYPE=AZCLI in ~/.bashrc.
- Kaggle commit runs keep going with the browser closed; check progress by searching the log for "all   588".

- Results: 54 epochs (4-hour cap). Test (2,936 images): P 0.912 / R 0.869 / mAP50 0.931 / mAP50-95 0.583
  (smoke test val baseline: P 0.735 / R 0.623 / mAP50 0.674 / mAP50-95 0.345).
- Logging: rewrote log_run_to_azure.py to batch metrics (the per-metric version took 10+ minutes for 54 epochs).

## Day 4 (Wed 11 Nov 2026): gap check and basket mining
- Hand-labelled 15 fresh test images (not used for tuning).
  Gap level: precision 53% (95% CI 41-65%), recall 52% (40-63%).
  Errors concentrated: 2 images give 60% of false gaps (precision 71% without them); 3 images give 69% of misses.
  Bay level (secondary view): 11 of 13 flagged bays truly had empty slots; all 11 bays with empty slots flagged;
  gap score vs empty slots Spearman 0.63 (p = 0.01).
- Basket mining (500k orders, FP-Growth): aisle rules show shopping occasions (dips -> chips, lift 2.2; lunch meat
  + yogurt -> cheese, conf 51%); top product rules by lift are flavour variants (Skyr, lift 74, 0.2% of orders),
  so judged by leverage as well as lift.
- Product metrics: fresh produce leads on reach, reorder rate and pull (Banana: 14.7% of orders, 85% reorders,
  59 partners).
- Simulated planogram: 60 aisles x 8 shelves x 5 positions, best sellers at eye level.

- Failure types (gap check): (1) fridge door frames not removed: the fixture rule counted all rows in the photo, and angled shots slant the frames; (2) products hidden by glare or shadow left undetected; (3) steep aisle perspective: distant products are missed and small empty slots fall under the row's median width; (4) promo signs in slots.
- v2 fixes planned for Day 6 (per-position row count, slanted alignment, neighbour-width threshold, low-confidence veto), evaluated on a second fresh set of 15 images.