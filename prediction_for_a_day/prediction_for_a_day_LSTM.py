import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt

# === CONFIG ===
DATA_PATH = '../electricity_cleaned_kWh.csv'
WEATHER_PATH = '../weather_Panther.csv'
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
    base_dir = os.path.dirname(os.path.realpath(__file__))
    target_date = pd.Timestamp(target_date)
    OUTPUT_DIR = f'prediction_for_{target_date.date()}_{target_date.strftime("%A")}'
    model_folder = os.path.join(base_dir, MODELS_DIR, building_id)
    output_folder = os.path.join(base_dir, OUTPUT_DIR, building_id)
    os.makedirs(output_folder, exist_ok=True)

    # Incarc datele de consum
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[['timestamp', building_id]].dropna()

    # Incarc datele de vreme
    weather_df = pd.read_csv(WEATHER_PATH)
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
    weather_df = weather_df[['timestamp', 'airTemperature', 'dewTemperature']]

    # Combin energie + vreme
    df = df.merge(weather_df, on='timestamp', how='left')
    df.set_index('timestamp', inplace=True)

    # Selectez intervalul necesar (cu 1 zi extra)
    start_date = target_date - pd.Timedelta(days=LOOKBACK_DAYS)
    df_subset = df.loc[start_date - pd.Timedelta(hours=24): target_date - pd.Timedelta(hours=1)].copy()

    if df_subset.empty:
        print("Nu sunt suficiente date pentru aceasta perioada.")
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

    # Extrag valorile de temperatura pentru cele 24 de ore
    weather_24h = weather_df.set_index('timestamp').loc[target_date:target_date + pd.Timedelta(hours=23)].copy()

    # Rezultate
    df_result = pd.DataFrame({
        'timestamp': timestamps,
        'hour': list(range(24)),
        'predicted_consumption': y_pred,
        'actual_consumption': [np.nan] * 24
    })

    if not weather_24h.empty and len(weather_24h) == 24:
        df_result['airTemperature'] = weather_24h['airTemperature'].values
        df_result['dewTemperature'] = weather_24h['dewTemperature'].values
    else:
        df_result['airTemperature'] = np.nan
        df_result['dewTemperature'] = np.nan
        print("[!] Nu am gasit temperaturi suficiente pentru cele 24 de ore.")

    # Incarc si compar cu valorile reale, daca exista
    try:
        real_df = pd.read_csv(DATA_PATH)
        real_df['timestamp'] = pd.to_datetime(real_df['timestamp'])
        real_day = real_df[['timestamp', building_id]]
        real_day = real_day.set_index('timestamp').loc[target_date:target_date + pd.Timedelta(hours=23)].copy()

        if not real_day.empty and len(real_day) == 24:
            df_result['actual_consumption'] = real_day[building_id].values

            # === GRAFIC COMPARATIV ===
            plt.figure(figsize=(12, 5))
            plt.plot(df_result['hour'], df_result['actual_consumption'], marker='o', color='blue', label='Consum real')
            plt.plot(df_result['hour'], df_result['predicted_consumption'], marker='o', color='green',
                     label='Consum Prezis')
            plt.title(f'Comparatie Real vs Predictie - {building_id} - {target_date.date()}')
            plt.xlabel('Ora (0-23)')
            plt.ylabel('Consum (kWh)')
            plt.grid(True)
            plt.legend()
            plt.xticks(np.arange(0, 24, 1))
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_folder, f'consumption_vs_prediction__{building_id}_{target_date.date()}.png'))
            plt.close()
    except Exception as e:
        print(f"[!] Nu am putut incarca consumul real: {e}")

    # === Calculez metrice doar daca am valori reale
    if df_result['actual_consumption'].notna().all():
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        y_true = df_result['actual_consumption'].values
        y_pred = df_result['predicted_consumption'].values

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

        df_result['mse'] = mse
        df_result['mae'] = mae
        df_result['r2'] = r2
        df_result['smape'] = smape
    else:
        print("[!] Nu am suficiente valori reale pentru a calcula metricile.")


    # === GRAFIC DOAR PREDICTIE ===
    plt.figure(figsize=(12, 5))
    plt.plot(df_result['hour'], df_result['predicted_consumption'], marker='o', color='green', label='Consum Prezis')
    plt.title(f'Consum prezis pentru {building_id} - {target_date.date()} ({day_name})')
    plt.xlabel('Ora (0-23)')
    plt.ylabel('Consum estimat (kWh)')
    plt.grid(True)
    plt.xticks(np.arange(0, 24, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'prediction_{building_id}_{target_date.date()}.png'))
    plt.close()

    # === Salvare CSV ===
    csv_path = os.path.join(output_folder, f'prediction_{building_id}_{target_date.date()}_{day_name}.csv')
    df_result.to_csv(csv_path, index=False)

    print(f"Predictia pentru {building_id} ({target_date.date()}) a fost salvata în {output_folder}")


# MAIN FUNCTION
if __name__ == '__main__':
    predict_energy_for_day('Panther_office_Ruthie', '2017-03-15')
