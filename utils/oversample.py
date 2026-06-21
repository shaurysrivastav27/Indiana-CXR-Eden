import pandas as pd


def balance_dataset(abnormal_X=3, reduce_normal=False):
    input_csv = "train_resized.csv"
    output_csv = "train_balanced.csv"

    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)

    # 1. Identify "Purely Normal" rows vs "Abnormal" rows
    # Assuming your 'buckets' column contains strings or lists of the target buckets
    # Adjust the string matching below based on exactly how your 'buckets' column is formatted
    def is_normal(bucket_val):
        val = str(bucket_val).lower()
        # A row is normal only if it contains "normal" or "no_finding"
        # and contains no other abnormal label.
        # We check this by ensuring no known abnormal bucket keyword is present.
        abnormal_keywords = [
            "cardiomegaly", "cardiovascular",
            "opacity", "infection", "consolidation",
            "pleural", "pneumothorax",
            "copd", "emphysema", "hyperinflation",
            "atelectasis", "volume_loss",
            "nodule", "mass", "fibrosis", "granuloma",
            "bones", "spine", "fracture",
            "devices", "surgical",
            "diaphragm", "mediastinum",
        ]
        has_normal = "normal" in val or "no_finding" in val
        has_abnormal = any(kw in val for kw in abnormal_keywords)
        return has_normal and not has_abnormal

    df["is_normal"] = df["buckets"].apply(is_normal)

    normal_df = df[df["is_normal"] == True]
    abnormal_df = df[df["is_normal"] == False]

    print(f"Original Normal Rows: {len(normal_df)}")
    print(f"Original Abnormal Rows: {len(abnormal_df)}")

    # 2. Undersample Normal (Keep only ~250)
    # This prevents the LLM from defaulting to normal text
    if reduce_normal:
        normal_downsampled = normal_df.sample(
            n=min(250, len(normal_df)), random_state=42
        )
    else:
        normal_downsampled = normal_df.copy()

    # 3. Oversample Abnormal
    # We will duplicate the abnormal rows to ensure the model sees enough disease examples
    # Duplicating it 3 times is usually a good starting point
    abnormal_upsampled = pd.concat([abnormal_df] * abnormal_X, ignore_index=True)

    # 4. Combine and Shuffle
    balanced_df = (
        pd.concat([normal_downsampled, abnormal_upsampled])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    # Drop the helper column
    balanced_df = balanced_df.drop(columns=["is_normal"])

    balanced_df.to_csv(output_csv, index=False)

    print(f"\n--- Balanced Dataset Stats ---")
    print(f"Normal Rows: {len(normal_downsampled)}")
    print(f"Abnormal Rows: {len(abnormal_upsampled)}")
    print(f"Total Rows: {len(balanced_df)}")
    print(f"Saved to {output_csv}")


if __name__ == "__main__":
    balance_dataset()
