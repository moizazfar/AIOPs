import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    Converts cleaned log rows into numerical feature vectors.

    WHY do we need this?
    Machine learning models cannot read text like "ERROR" or "timeout".
    They only understand NUMBERS.

    So we take all log lines in a 1-minute window and ask:
    - How many errors in this minute?
    - What % of logs were errors?
    - Was there a CPU spike?
    - Were there any crashes?
    ...and so on.

    Each 1-minute window becomes ONE row with all these numbers.
    That row is what the ML model gets.
    """

    def extract(self, df: pd.DataFrame, window_minutes: int = 1) -> pd.DataFrame:
        """
        Aggregate logs per time window and extract features.

        Input:  DataFrame where each row = one log line
        Output: DataFrame where each row = one time window (1 min)
                with numerical features

        window_minutes: how long each time bucket is
        """
        df = df.copy()

        # Ensure timestamp is the index for resampling
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp').sort_index()

        print(f"\n--- FEATURE ENGINEERING ---")
        print(f"Window size: {window_minutes} minute(s)")
        print(f"Input rows: {len(df)}")

        windows = []  # list of feature dicts, one per window

        # resample groups rows into time buckets
        # '1min' = one bucket per minute
        for window_start, window_df in df.resample(f'{window_minutes}min'):

            # Skip empty windows (no logs in this minute)
            if len(window_df) == 0:
                continue

            total_lines = len(window_df)

            # ── Count features ────────────────────────────
            # How many logs of each severity level?
            level_counts = {}
            if 'log.level' in window_df.columns:
                vc = window_df['log.level'].str.upper().value_counts()
                level_counts = {
                    'error_count':    int(vc.get('ERROR',    0)),
                    'warn_count':     int(vc.get('WARN',     0) + vc.get('WARNING', 0)),
                    'info_count':     int(vc.get('INFO',     0)),
                    'critical_count': int(vc.get('CRITICAL', 0)),
                }
            else:
                level_counts = {
                    'error_count': 0, 'warn_count': 0,
                    'info_count': 0, 'critical_count': 0
                }

            # ── Rate features ─────────────────────────────
            # What PERCENTAGE of logs were errors?
            error_rate = level_counts['error_count'] / total_lines * 100
            warn_rate  = level_counts['warn_count']  / total_lines * 100

            # ── Keyword features ──────────────────────────
            # Search for specific dangerous words in log messages
            keyword_counts = {}
            if 'log.message' in window_df.columns:
                messages = window_df['log.message'].astype(str).str.lower()
                keyword_counts = {
                    'timeout_count':  int(messages.str.contains('timeout').sum()),
                    'crash_count':    int(messages.str.contains('crash|crashed|fatal').sum()),
                    'db_error_count': int(messages.str.contains('database|connection refused').sum()),
                    'memory_count':   int(messages.str.contains('memory|oom').sum()),
                    'restart_count':  int(messages.str.contains('restart|restarting').sum()),
                }
            else:
                keyword_counts = {k: 0 for k in [
                    'timeout_count', 'crash_count', 'db_error_count',
                    'memory_count', 'restart_count'
                ]}

            # ── Metric features ───────────────────────────
            # Average CPU and memory in this window
            metric_feats = {}
            if 'metrics.cpu_percent' in window_df.columns:
                cpu = window_df['metrics.cpu_percent'].dropna()
                metric_feats['cpu_mean']   = round(float(cpu.mean()), 2) if len(cpu) > 0 else 0.0
                metric_feats['cpu_max']    = round(float(cpu.max()),  2) if len(cpu) > 0 else 0.0

            if 'metrics.memory_percent' in window_df.columns:
                mem = window_df['metrics.memory_percent'].dropna()
                metric_feats['memory_mean']= round(float(mem.mean()), 2) if len(mem) > 0 else 0.0
                metric_feats['memory_max'] = round(float(mem.max()),  2) if len(mem) > 0 else 0.0

            # ── Derived features ──────────────────────────
            # Computed FROM other features for richer signal
            error_to_info = level_counts['error_count'] / (level_counts['info_count'] + 1)
            resource_pressure = (
                metric_feats.get('cpu_mean', 0) +
                metric_feats.get('memory_mean', 0)
            ) / 2

            # ── Time features ─────────────────────────────
            time_feats = {
                'hour':         window_start.hour,
                'is_weekend':   int(window_start.dayofweek >= 5),
                'is_peak_hour': int(9 <= window_start.hour <= 17),
            }

            # ── Combine all features into one dict ────────
            feature_row = {
                'window_start':      window_start,
                'total_lines':       total_lines,
                **level_counts,
                'error_rate':        round(error_rate, 2),
                'warn_rate':         round(warn_rate, 2),
                **keyword_counts,
                **metric_feats,
                'error_to_info':     round(error_to_info, 3),
                'resource_pressure': round(resource_pressure, 2),
                **time_feats,
            }
            windows.append(feature_row)

        feature_df = pd.DataFrame(windows).fillna(0)

        print(f"Output windows: {len(feature_df)}")
        print(f"Features per window: {len(feature_df.columns) - 1}")  # -1 for window_start
        print(f"\nFeature preview:")
        print(feature_df.to_string())

        return feature_df
