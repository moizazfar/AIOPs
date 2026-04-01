import pandas as pd
import numpy as np


class Preprocessor:
    """
    Cleans the raw flattened DataFrame.

    Problems we fix:
    1. Missing values (NaN)       → fill with 0 or empty string
    2. Duplicate log lines        → remove exact duplicates
    3. Noise (health checks etc.) → remove useless lines
    4. Wrong data types           → fix numeric columns stored as strings
    """

    # Patterns we want to IGNORE because they are not real issues
    NOISE_KEYWORDS = [
        'health check', 'healthz', 'liveness probe',
        'readiness probe', 'ping', '/metrics', '/ready'
    ]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline."""
        print("\n--- PREPROCESSING ---")
        original_count = len(df)

        # Step 1: Remove rows with no timestamp
        df = df.dropna(subset=['timestamp'])
        print(f"After timestamp filter: {len(df)} rows")

        # Step 2: Remove duplicate rows
        # Keep first occurrence, drop exact duplicates
        df = df.drop_duplicates()
        print(f"After deduplication:    {len(df)} rows")

        # Step 3: Remove noise (health checks, probes)
        if 'log.message' in df.columns:
            # Create a mask: True = row is noise
            noise_mask = df['log.message'].astype(str).str.lower().apply(
                lambda msg: any(kw in msg for kw in self.NOISE_KEYWORDS)
            )
            # Keep rows where noise_mask is False
            df = df[~noise_mask]
            print(f"After noise removal:    {len(df)} rows")

        # Step 4: Fix data types
        # Convert metric columns to numeric (they might be stored as strings)
        numeric_cols = ['metrics.cpu_percent', 'metrics.memory_percent']
        for col in numeric_cols:
            if col in df.columns:
                # errors='coerce' → invalid values become NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Step 5: Fill missing values
        # Numeric columns → 0
        num_cols = df.select_dtypes(include=['number']).columns
        df[num_cols] = df[num_cols].fillna(0)

        # String columns → empty string
        str_cols = df.select_dtypes(include=['object']).columns
        df[str_cols] = df[str_cols].fillna('')

        # Step 6: Clip extreme values (e.g. CPU cannot be > 100%)
        for col in ['metrics.cpu_percent', 'metrics.memory_percent']:
            if col in df.columns:
                df[col] = df[col].clip(0, 100)

        removed = original_count - len(df)
        print(f"\nCleaning summary:")
        print(f"  Original rows:  {original_count}")
        print(f"  Removed rows:   {removed}")
        print(f"  Remaining rows: {len(df)}")

        return df.reset_index(drop=True)

    def show_stats(self, df: pd.DataFrame):
        """Print basic statistics about the cleaned data."""
        print("\n--- DATA STATISTICS ---")

        if 'log.level' in df.columns:
            print("\nLog level distribution:")
            print(df['log.level'].value_counts().to_string())

        if 'metrics.cpu_percent' in df.columns:
            print(f"\nCPU percent stats:")
            print(df['metrics.cpu_percent'].describe().round(2).to_string())

        if 'metrics.memory_percent' in df.columns:
            print(f"\nMemory percent stats:")
            print(df['metrics.memory_percent'].describe().round(2).to_string())
