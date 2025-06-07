import pandas as pd
import joblib
import os

MODEL_DIR = "model/"

# Load kolom terpilih
selected_columns = joblib.load(os.path.join(MODEL_DIR, "selected_columns.pkl"))

def preprocess_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mengambil hanya kolom-kolom yang digunakan dalam model.

    Args:
        df (pd.DataFrame): Data input.

    Returns:
        pd.DataFrame: Data yang sudah difilter hanya berisi kolom yang dipakai model.
    """
    df = df.copy()

    # Validasi kolom
    missing_cols = [col for col in selected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom berikut tidak ditemukan di data: {missing_cols}")

    # Ambil kolom yang dibutuhkan
    X = df[selected_columns]

    return X
