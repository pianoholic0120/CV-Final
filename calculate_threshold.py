import pandas as pd
import numpy as np
import sys
import os

def compute_threshold_mad(scores, factor=2):
    """
    Calculate threshold using median and MAD:
      Threshold = median - factor * MAD
    """
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    threshold = median - factor * mad
    return max(threshold, 0)

def main(input_csv):
    df = pd.read_csv(input_csv)
    if "Score" not in df.columns:
        raise ValueError("CSV needs a column named 'Score'")
    
    threshold = compute_threshold_mad(df["Score"], factor=2) # Smaller factor will preserve more images
    print(f"Calculated Threshold: {threshold:.4f}")

    df_filtered = df[df["Score"] >= threshold].copy()

    base, ext = os.path.splitext(input_csv)
    output_csv = base + "_filtered" + ext

    df_filtered.to_csv(output_csv, index=False)
    print(f"Filtered data information has been saved in {output_csv}")

if __name__ == "__main__":
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "input.csv"
    main(input_csv)
