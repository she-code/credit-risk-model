import pandas as pd
import joblib
from pipeline import build_pipeline
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH, MODEL_ARTIFACT_PATH, TARGET_COL


def main():
    # Load raw data
    df = pd.read_csv(RAW_DATA_PATH)

    # Separate features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Build and fit pipeline
    pipeline = build_pipeline()
    X_processed = pipeline.fit_transform(X)

    # Combine with target
    processed_df = pd.DataFrame(
        X_processed,
        columns=pipeline.named_steps["preprocessor"].get_feature_names_out(),
    )
    processed_df[TARGET_COL] = y.values

    # Save processed data and pipeline
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    joblib.dump(pipeline, MODEL_ARTIFACT_PATH)

    print("Pipeline executed successfully!")
    print(f"Processed data saved to: {PROCESSED_DATA_PATH}")
    print(f"Pipeline artifact saved to: {MODEL_ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
