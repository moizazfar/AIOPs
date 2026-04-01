import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# These are the columns we feed into the model
# window_start is metadata, not a feature
FEATURE_COLS = [
    'total_lines',
    'error_count', 'warn_count', 'info_count',
    'error_rate', 'warn_rate',
    'timeout_count', 'crash_count', 'db_error_count',
    'memory_count', 'restart_count',
    'cpu_mean', 'cpu_max',
    'memory_mean', 'memory_max',
    'error_to_info', 'resource_pressure',
    'hour', 'is_weekend', 'is_peak_hour',
]


class AnomalyDetector:
    """
    Isolation Forest-based anomaly detector.

    Simple 3-step process:
    1. train(data)    → learns what normal looks like
    2. predict(data)  → scores each window
    3. report(data)   → builds a readable results table
    """

    def __init__(self, contamination: float = 0.1):
        """
        contamination = how much of our data we expect to be anomalous
        0.1 = we expect 10% of windows to be anomalies
        """
        self.contamination = contamination
        self.model  = None
        self.scaler = None

    def _get_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract only the feature columns from the DataFrame.
        Any column not in FEATURE_COLS is ignored.
        Missing columns are filled with 0.
        """
        X = pd.DataFrame(index=df.index)
        for col in FEATURE_COLS:
            X[col] = df[col] if col in df.columns else 0.0
        return X.values.astype(np.float64)

    def train(self, feature_df: pd.DataFrame):
        """
        Train the Isolation Forest on historical data.

        Steps:
        1. Extract feature matrix X
        2. Scale: make all features have mean=0, std=1
           Why? Error count might be 0-30, CPU might be 0-100.
           Without scaling, CPU would dominate. Scaling equalizes them.
        3. Train: build 200 random trees
        """
        print("\n--- MODEL TRAINING ---")
        X = self._get_features(feature_df)
        print(f"Training on {X.shape[0]} windows x {X.shape[1]} features")

        # Scale features
        self.scaler = StandardScaler()
        X_scaled    = self.scaler.fit_transform(X)

        # Train Isolation Forest
        self.model = IsolationForest(
            n_estimators  = 200,               # number of trees
            contamination = self.contamination, # expected anomaly %
            max_samples   = 'auto',
            random_state  = 42,
            n_jobs        = -1                  # use all CPU cores
        )
        self.model.fit(X_scaled)

        print(f"Model trained with {self.model.n_estimators} trees")
        print(f"Expected anomaly rate: {self.contamination * 100}%")

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score each window and flag anomalies.

        Returns feature_df with 3 new columns added:
        - anomaly_score:  more negative = more anomalous
        - is_anomaly:     True / False
        - severity:       normal / high / critical
        """
        if self.model is None:
            raise RuntimeError("Model not trained! Call train() first.")

        X        = self._get_features(feature_df)
        X_scaled = self.scaler.transform(X)  # transform ONLY, do NOT fit again

        # score_samples: more negative = more anomalous
        scores      = self.model.score_samples(X_scaled)
        # predict: -1 = anomaly, +1 = normal
        predictions = self.model.predict(X_scaled)

        result = feature_df.copy()
        result['anomaly_score'] = scores.round(4)
        result['is_anomaly']    = predictions == -1

        # Assign severity label
        def severity(score):
            if score < -0.4:
                return 'critical'
            elif score < -0.25:
                return 'high'
            else:
                return 'normal'

        result['severity'] = result['anomaly_score'].apply(severity)

        anomaly_count = result['is_anomaly'].sum()
        print(f"\n--- PREDICTION RESULTS ---")
        print(f"Total windows:    {len(result)}")
        print(f"Anomalies found:  {anomaly_count}")
        print(f"Normal windows:   {len(result) - anomaly_count}")

        return result

    def report(self, result_df: pd.DataFrame):
        """
        Print a clean anomaly report.
        Shows details for each detected anomaly.
        """
        print("\n" + "=" * 60)
        print("ANOMALY DETECTION REPORT")
        print("=" * 60)

        anomalies = result_df[result_df['is_anomaly']].copy()
        anomalies = anomalies.sort_values('anomaly_score')  # worst first

        if len(anomalies) == 0:
            print("No anomalies detected. System looks healthy!")
            return

        print(f"Total anomalies detected: {len(anomalies)}\n")

        for _, row in anomalies.iterrows():
            print(f"Time:     {row['window_start']}")
            print(f"Score:    {row['anomaly_score']}  (lower = worse)")
            print(f"Severity: {row['severity'].upper()}")
            print(f"Logs:     {int(row['total_lines'])} total | "
                  f"{int(row.get('error_count', 0))} errors | "
                  f"{row.get('error_rate', 0):.1f}% error rate")
            if row.get('crash_count', 0) > 0:
                print(f"CRASHES:  {int(row['crash_count'])} crash events!")
            if row.get('db_error_count', 0) > 0:
                print(f"DB ERR:   {int(row['db_error_count'])} database errors")
            print(f"CPU:      avg={row.get('cpu_mean', 0):.1f}%  "
                  f"max={row.get('cpu_max', 0):.1f}%")
            print(f"Memory:   avg={row.get('memory_mean', 0):.1f}%  "
                  f"max={row.get('memory_max', 0):.1f}%")
            print("-" * 40)

    def plot(self, result_df: pd.DataFrame, save_path: str = 'output/results.png'):
        """
        Create 3 charts:
        1. Error rate over time — with anomalies highlighted in red
        2. CPU usage over time
        3. Anomaly scores over time
        """
        import os
        os.makedirs('output', exist_ok=True)

        normal    = result_df[~result_df['is_anomaly']]
        anomalies = result_df[result_df['is_anomaly']]

        fig, axes = plt.subplots(3, 1, figsize=(12, 9))
        fig.suptitle('AIOps Anomaly Detection Results', fontsize=14, fontweight='bold')

        # Chart 1: Error rate
        ax1 = axes[0]
        ax1.plot(normal['window_start'],    normal['error_rate'],
                 'o-', color='steelblue', label='Normal', linewidth=1.5)
        ax1.scatter(anomalies['window_start'], anomalies['error_rate'],
                    color='red', s=120, zorder=5, label='Anomaly', marker='v')
        ax1.set_ylabel('Error rate (%)')
        ax1.set_title('Error rate over time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Chart 2: CPU usage
        ax2 = axes[1]
        ax2.plot(normal['window_start'],    normal['cpu_mean'],
                 'o-', color='green', label='Normal CPU', linewidth=1.5)
        ax2.scatter(anomalies['window_start'], anomalies['cpu_mean'],
                    color='red', s=120, zorder=5, label='Anomaly CPU', marker='v')
        ax2.set_ylabel('CPU (%)')
        ax2.set_title('CPU usage over time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Chart 3: Anomaly scores
        ax3 = axes[2]
        colors = ['red' if a else 'steelblue' for a in result_df['is_anomaly']]
        ax3.bar(range(len(result_df)), result_df['anomaly_score'], color=colors)
        ax3.axhline(y=-0.25, color='orange', linestyle='--', label='High threshold')
        ax3.axhline(y=-0.40, color='red',    linestyle='--', label='Critical threshold')
        ax3.set_ylabel('Anomaly score')
        ax3.set_xlabel('Window index')
        ax3.set_title('Anomaly scores (red bars = detected anomalies)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nChart saved to: {save_path}")
