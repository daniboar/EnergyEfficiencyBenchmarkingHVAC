import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch
import torch.nn as nn
import joblib
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Incarc datele din csv
data = pd.read_csv('electricity_cleaned_kWh.csv')
data = data.iloc[:, :31]  # Timestamp + primele 30 de cladiri
output_folder = 'lstm_daily_predictions'
os.makedirs(output_folder, exist_ok=True)

# Parametrii modelului
LOOKBACK_DAYS = 3  # Folosesc ultimele 3 zile (3 * 24 ore) pentru predicție
PREDICTION_HORIZON = 24  # Vreau sa prezic pentru urmatoareal 24 de ore

metrics_log = []

# 2. Model LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


# 3. Dataset pentru PyTorch
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.astype(np.float32), dtype=torch.float32)
        self.y = torch.tensor(y.astype(np.float32), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 4. Functie pentru caracteristicile temporale
def create_time_series_features(df, target_column, lookback_days):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Caracteristici temporale
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week
    df['weekday'] = df.index.weekday
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)

    # Anotimpuri
    df['season'] = df['month'].apply(
        lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4)

    # One-hot encoding pentru ziua saptamanii si sezon
    df = pd.get_dummies(df, columns=['weekday', 'season'])

    # Lag-uri pentru ultimele `lookback_days` zile (fiecare zi are 24 ore)
    for lag in range(1, lookback_days * 24 + 1):
        df[f'lag_{lag}'] = df[target_column].shift(lag)

    df.dropna(inplace=True)
    return df

# 5. Iterez prin cele 30 de cladiri si antrenez modelul
for building_id in tqdm(data.columns[1:31], desc="Procesare cladiri"):
    print(f"\nProcesare pentru cladirea: {building_id}")

    building_data = data[['timestamp', building_id]].dropna()

    # Creez setul de date cu caracteristici temporale
    building_data = create_time_series_features(building_data, building_id, LOOKBACK_DAYS)

    # Salvam separat target-ul si features
    feature_data = building_data.drop(columns=[building_id])
    target_data = building_data[building_id]

    # Normalizez X si y
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(feature_data)
    y = scaler_y.fit_transform(target_data.values.reshape(-1, 1))

    # Construiesc X si y secvențial pentru predicție pe 24 ore
    X_seq, y_seq, timestamps = [], [], []
    for i in range(len(X) - PREDICTION_HORIZON):
        X_seq.append(X[i])
        y_seq.append(y[i + 1: i + 1 + PREDICTION_HORIZON].flatten())
        timestamps.append(building_data.index[i + 1: i + 1 + PREDICTION_HORIZON].tolist())

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Train/Val/Test split
    X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=64, shuffle=False)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=64, shuffle=False)

    # Model
    input_dim = X.shape[1]
    model = LSTMModel(input_dim=input_dim, hidden_dim=256, output_dim=PREDICTION_HORIZON, num_layers=3)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Antrenare
    for epoch in range(200):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/200 - Train Loss: {total_loss / len(train_loader):.4f}")

    # Salvare model si scalere
    model_folder = os.path.join('modele salvate LSTM', building_id)
    os.makedirs(model_folder, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_folder, 'lstm_model.pt'))
    joblib.dump(scaler_X, os.path.join(model_folder, 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(model_folder, 'scaler_y.pkl'))

    with open(os.path.join(model_folder, 'features.txt'), 'w') as f:
        for col in feature_data.columns:
            f.write(f"{col}\n")

    print(f"Model salvat pentru {building_id} in {model_folder}")

    # Evaluare
    model.eval()
    y_test_pred = []
    y_test_actual = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            y_test_pred.append(y_pred.numpy())
            y_test_actual.append(y_batch.numpy())

    y_test_pred = np.concatenate(y_test_pred)
    y_test_actual = np.concatenate(y_test_actual)
    y_test_pred = scaler_y.inverse_transform(y_test_pred)
    y_test_actual = scaler_y.inverse_transform(y_test_actual)

    mse = mean_squared_error(y_test_actual, y_test_pred)
    mae = mean_absolute_error(y_test_actual, y_test_pred)
    r2 = r2_score(y_test_actual, y_test_pred)
    smape = np.mean(2 * np.abs(y_test_pred - y_test_actual) / (np.abs(y_test_pred) + np.abs(y_test_actual))) * 100

    metrics_log.append([building_id, mse, mae, r2, smape])
    print(f"{building_id} | MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}, SMAPE: {smape:.2f}%")

# Salvare metrici
metrics_df = pd.DataFrame(metrics_log, columns=['Building', 'MSE', 'MAE', 'R2', 'SMAPE'])
metrics_df.to_csv('lstm_metrics.csv', index=False)

print("\nToate modelele au fost antrenate si salvate corect!")
