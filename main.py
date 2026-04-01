# ════════════════════════════════════════════════════════════
# main.py
# Runs the complete pipeline:
# Load → Flatten → Clean → Extract Features → Train → Detect
# ════════════════════════════════════════════════════════════

import os
import pandas as pd

from src.data_loader      import load_json_logs, logs_to_dataframe
from src.preprocessor     import Preprocessor
from src.feature_engineer import FeatureEngineer
from src.anomaly_model    import AnomalyDetector

os.makedirs('output', exist_ok=True)

print("=" * 60)
print("AIOps ANOMALY DETECTION PIPELINE")
print("=" * 60)

# ── STEP 1: Load raw JSON logs ───────────────────────────────
print("\n[STEP 1] Loading raw JSON logs...")
raw_records = load_json_logs('data/sample_logs.json')
raw_df      = logs_to_dataframe(raw_records)

# Save raw flattened data (to show LM what flattening does)
raw_df.to_csv('output/01_raw_flattened.csv', index=False)
print(f"Saved: output/01_raw_flattened.csv")

# ── STEP 2: Clean and preprocess ────────────────────────────
print("\n[STEP 2] Cleaning and preprocessing...")
preprocessor = Preprocessor()
clean_df     = preprocessor.clean(raw_df)
preprocessor.show_stats(clean_df)

# Save cleaned data
clean_df.to_csv('output/02_cleaned.csv', index=False)
print(f"Saved: output/02_cleaned.csv")

# ── STEP 3: Extract features ─────────────────────────────────
print("\n[STEP 3] Extracting features...")
fe         = FeatureEngineer()
feature_df = fe.extract(clean_df, window_minutes=1)

# Save feature matrix
feature_df.to_csv('output/03_features.csv', index=False)
print(f"Saved: output/03_features.csv")

# ── STEP 4: Train model and detect anomalies ─────────────────
print("\n[STEP 4] Training Isolation Forest model...")
detector = AnomalyDetector(contamination=0.15)
detector.train(feature_df)

print("\n[STEP 5] Detecting anomalies...")
result_df = detector.predict(feature_df)

# ── STEP 6: Report and visualize ─────────────────────────────
detector.report(result_df)
detector.plot(result_df, save_path='output/04_results.png')

# Save final report
report_cols = [
    'window_start', 'total_lines', 'error_count',
    'error_rate', 'cpu_mean', 'memory_mean',
    'anomaly_score', 'is_anomaly', 'severity'
]
available = [c for c in report_cols if c in result_df.columns]
result_df[available].to_csv('output/05_anomaly_report.csv', index=False)
print(f"Saved: output/05_anomaly_report.csv")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print("\nOutput files:")
print("  output/01_raw_flattened.csv  → after JSON flattening")
print("  output/02_cleaned.csv        → after preprocessing")
print("  output/03_features.csv       → ML-ready feature matrix")
print("  output/04_results.png        → charts")
print("  output/05_anomaly_report.csv → final anomaly report")
