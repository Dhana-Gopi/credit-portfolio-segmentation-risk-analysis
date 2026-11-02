
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"

prep = joblib.load(
    MODELS_DIR / "final_preprocessing.joblib"
)

def preprocess_borrowers(raw_df, prep_bundle=prep):

    required_columns = [
        "revenue",
        "dti_n",
        "loan_amnt",
        "fico_n",
        "emp_length",
        "home_ownership_n",
        "purpose"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in raw_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = raw_df.copy()

    # Convert numerical input fields
    for col in [
        "revenue",
        "dti_n",
        "loan_amnt",
        "fico_n"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise"
        )

    # Check for missing or invalid numerical values
    numeric_cols = [
        "revenue", "dti_n", "loan_amnt", "fico_n"
    ]

    if not np.isfinite(df[numeric_cols].to_numpy()).all():
        raise ValueError(
            "Numerical inputs must be finite and non-missing."
        )

    if (
        (df["revenue"] <= 0).any()
        or (df["loan_amnt"] <= 0).any()
        or (df["dti_n"] < 0).any()
        or (~df["fico_n"].between(300, 850)).any()
    ):
        raise ValueError(
            "Invalid income, loan amount, DTI, or FICO value."
        )

    # Income transformation
    df["Revenue_Log_Final"] = np.log1p(
        df["revenue"]
    )

    # DTI upper capping
    df["DTI_IQR_Final"] = df["dti_n"].clip(
        upper=prep_bundle["dti_cap"]
    )

    # Employment missing-value flag
    df["Employment_NI_Flag"] = (
        df["emp_length"] == "NI"
    ).astype(int)

    # Employment years conversion
    employment_mapping = prep_bundle[
        "employment_mapping"
    ]

    unknown_employment = (
        set(df["emp_length"].dropna())
        - set(employment_mapping)
    )

    if unknown_employment or df["emp_length"].isna().any():
        raise ValueError(
            f"Unknown or missing employment categories: "
            f"{unknown_employment}"
        )

    df["Employment_Years_Imputed"] = (
        df["emp_length"]
        .map(employment_mapping)
        .fillna(prep_bundle["employment_median"])
        .astype(float)
    )

    # Validate categorical inputs
    for col in prep_bundle["categorical_columns"]:
        if df[col].isna().any():
            raise ValueError(
                f"Missing categorical values in {col}"
            )

        allowed = set(
            prep_bundle["encoder"].categories_[
                prep_bundle["categorical_columns"].index(col)
            ]
        )

        unknown = set(df[col]) - allowed

        if unknown:
            raise ValueError(
                f"Unknown categories in {col}: {unknown}"
            )

    # Scale numerical features
    numerical_cols = prep_bundle["numerical_columns"]

    scaled_values = prep_bundle["scaler"].transform(
        df[numerical_cols]
    )

    scaled_df = pd.DataFrame(
        scaled_values,
        columns=numerical_cols,
        index=df.index
    )

    # One-hot encode categorical features
    categorical_cols = prep_bundle["categorical_columns"]

    encoded_values = prep_bundle["encoder"].transform(
        df[categorical_cols]
    )

    if hasattr(encoded_values, "toarray"):
        encoded_values = encoded_values.toarray()

    encoded_df = pd.DataFrame(
        encoded_values,
        columns=prep_bundle["encoder"].get_feature_names_out(),
        index=df.index
    )

    # Assemble all 24 features
    final_features = pd.concat(
        [
            scaled_df,
            df[["Employment_NI_Flag"]],
            encoded_df
        ],
        axis=1
    )

    final_features = final_features[
        prep_bundle["feature_columns"]
    ]

    return final_features
