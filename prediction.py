import pandas as pd
import pickle
import os
from data_preprocessing import preprocess_input_data

# Load model dan mapping saat modul diimpor
MODEL_PATH = os.path.join('model', 'best_model.pkl')
MAPPING_PATH = os.path.join('model', 'mapping_status.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(MAPPING_PATH, 'rb') as f:
    mapping_status = pickle.load(f)

# Reverse mapping
reverse_mapping = {v: k for k, v in mapping_status.items()}

def make_prediction(data: pd.DataFrame):
    """
    Lakukan prediksi pada DataFrame input dan kembalikan label + probabilitas.

    Args:
        data (pd.DataFrame): Data yang akan diprediksi

    Returns:
        tuple: (label hasil prediksi, probabilitas prediksi)
    """
    # Preprocessing
    X = preprocess_input_data(data)

    # Prediksi
    probas = model.predict_proba(X)
    preds = model.predict(X)
    labels = [reverse_mapping.get(p, 'Unknown') for p in preds]

    return labels, probas

# Jika file ini dijalankan langsung (bukan diimpor)
if __name__ == "__main__":
    # Load data uji
    data = pd.read_csv('dataset/data_test.csv', delimiter=';')

    # Lakukan prediksi
    pred_labels, probas = make_prediction(data)

    # Simpan hasil ke CSV
    hasil = pd.DataFrame({
        'Predicted_Status': pred_labels,
        'Prob_Tidak_Dropout': probas[:, 0],
        'Prob_Dropout': probas[:, 1]
    })
    hasil.to_csv('dataset/hasil_prediksi_dropout.csv', index=False)

    print("Prediksi selesai. Hasil disimpan ke 'dataset/hasil_prediksi_dropout.csv'")