import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt

# === CONFIG ===
DATA_PATH = '../electricity_cleaned_kWh.csv'
MODELS_DIR = '../electricity_prediction_daily/modele salvate LSTM'
LOOKBACK_DAYS = 3
PREDICTION_HORIZON = 24


# === MODEL DEFINITIE ===
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# === FEATURE ENGINEERING ===
def create_features(df, target_column):
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week
    df['weekday'] = df.index.weekday
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    df['season'] = df['month'].apply(
        lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4)
    df = pd.get_dummies(df, columns=['weekday', 'season'])

    for lag in range(1, LOOKBACK_DAYS * 24 + 1):
        df[f'lag_{lag}'] = df[target_column].shift(lag)

    df.dropna(inplace=True)
    return df


# === FUNCTIA PRINCIPALA ===
def predict_energy_for_day(building_id: str, target_date: str):
    target_date = pd.Timestamp(target_date)
    OUTPUT_DIR = f'prediction_for_{target_date.date()}_{target_date.strftime("%A")}'
    model_folder = os.path.join(MODELS_DIR, building_id)
    output_folder = os.path.join(OUTPUT_DIR, building_id)
    os.makedirs(output_folder, exist_ok=True)

    # Incarc datele
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[['timestamp', building_id]].dropna()
    df.set_index('timestamp', inplace=True)

    # Selectez intervalul necesar (cu 1 zi extra)
    start_date = target_date - pd.Timedelta(days=LOOKBACK_DAYS)
    df_subset = df.loc[start_date - pd.Timedelta(hours=24): target_date - pd.Timedelta(hours=1)].copy()

    if df_subset.empty:
        print("Nu sunt suficiente date pentru această perioad.")
        return

    # Feature engineering
    df_feat = create_features(df_subset.copy(), building_id)

    # Incarc scalerele si feature list
    scaler_X = joblib.load(os.path.join(model_folder, 'scaler_X.pkl'))
    scaler_y = joblib.load(os.path.join(model_folder, 'scaler_y.pkl'))
    with open(os.path.join(model_folder, 'features.txt')) as f:
        saved_features = [line.strip() for line in f.readlines()]

    # Completez coloanele lipsa cu 0
    for col in saved_features:
        if col not in df_feat.columns:
            df_feat[col] = 0
    df_feat = df_feat[saved_features]

    # Scalez input-ul
    X_scaled = scaler_X.transform(df_feat)
    X_input = torch.tensor(X_scaled[-1].astype(np.float32)).unsqueeze(0)

    # Incarc modelul
    input_dim = X_scaled.shape[1]
    model = LSTMModel(input_dim, hidden_dim=256, output_dim=PREDICTION_HORIZON, num_layers=3)
    model.load_state_dict(torch.load(os.path.join(model_folder, 'lstm_model.pt'), weights_only=True))
    model.eval()

    # Predictie
    with torch.no_grad():
        y_pred_scaled = model(X_input).numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

    # Generez timestamp-urile
    timestamps = pd.date_range(start=target_date, periods=24, freq='h')
    day_name = target_date.strftime('%A')

    # Rezultate
    df_result = pd.DataFrame({
        'hour': list(range(24)),
        'timestamp': timestamps,
        'predicted_consumption': y_pred
    })

    # Salvare CSV + Plot
    csv_path = os.path.join(output_folder, f'prediction_{building_id}_{target_date.date()}_{day_name}.csv')
    png_path = os.path.join(output_folder, f'prediction_{building_id}_{target_date.date()}_{day_name}.png')

    df_result.to_csv(csv_path, index=False)

    plt.figure(figsize=(12, 5))
    plt.plot(df_result['hour'], df_result['predicted_consumption'], marker='o', color='green')
    plt.title(f'Predictie pentru {building_id} - {target_date.date()} ({day_name})')
    plt.xlabel('Ora (0-23)')
    plt.ylabel('Consum estimat (kWh)')
    plt.grid(True)
    plt.xticks(np.arange(0, 24, 1))
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    print(f"Predictia pentru {building_id} ({target_date.date()}) salvata în {output_folder}")


# MAIN FUNCTION
if __name__ == '__main__':
    predict_energy_for_day('Panther_parking_Lorriane', '2018-01-01')
