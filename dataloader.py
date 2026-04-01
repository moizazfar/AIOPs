import json
import pandas as pd


def load_json_logs(filepath: str) -> list:
    """
    Load raw JSON log file.
    Returns list of log dictionaries.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} raw log records from {filepath}")
    return data


def flatten_record(record: dict, parent_key: str = '', sep: str = '.') -> dict:
    """
    Flatten a nested dictionary into a single-level dict.

    Example:
        Input:  {"log": {"level": "ERROR", "message": "timeout"}}
        Output: {"log.level": "ERROR", "log.message": "timeout"}

    Why we need this:
        Real logs from Kubernetes/Fluentd are deeply nested JSON.
        pandas cannot work with nested dicts directly.
        We need flat key-value pairs to build a DataFrame.
    """
    flat = {}
    for key, value in record.items():
        # Build the full dotted key path
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            # Recurse into nested dict
            flat.update(flatten_record(value, new_key, sep))
        elif isinstance(value, list):
            # Convert list to string
            flat[new_key] = str(value)
        else:
            flat[new_key] = value

    return flat


def logs_to_dataframe(records: list) -> pd.DataFrame:
    """
    Convert list of raw log records into a pandas DataFrame.
    Each record is flattened first.

    Steps:
    1. Flatten each nested record
    2. Build DataFrame (each record = one row)
    3. Parse timestamp column
    4. Sort by time
    """
    # Flatten every record
    flattened = [flatten_record(r) for r in records]

    # Build DataFrame
    df = pd.DataFrame(flattened)

    # Find and parse timestamp column
    # Different log sources use different timestamp field names
    for ts_col in ['@timestamp', 'timestamp', 'time', 'event_time']:
        if ts_col in df.columns:
            df['timestamp'] = pd.to_datetime(df[ts_col], utc=True)
            break

    # Sort chronologically
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"\nDataFrame created: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    return df
