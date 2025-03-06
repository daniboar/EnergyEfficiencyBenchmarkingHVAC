import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Incarc datele din csv
data = pd.read_csv('electricity_cleaned.csv')
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
    df['season'] = df['month'].apply(lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4)

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

    # Construiesc X (input) și y (output)
    X, y, timestamps = [], [], []
    for i in range(len(building_data) - PREDICTION_HORIZON):
        X.append(building_data.iloc[i].values)
        y.append(building_data[building_id].iloc[i + 1: i + 1 + PREDICTION_HORIZON].values)

        # Salvez toate cele 24 de timestamp-uri asociate predicției
        timestamps.append(building_data.index[i + 1: i + 1 + PREDICTION_HORIZON].tolist())

    # Adaug timestamp-urile intr-o singura lista (flatten)
    timestamps = [ts for sublist in timestamps for ts in sublist]

    X = np.array(X)
    y = np.array(y)

    # Normalizez X si y cu MinMaxScaler
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    # Impart datele în train (80%), validare (10%), test (10%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

    # Creez dataseturile pentru PyTorch
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Construiesc modelul LSTM
    input_dim = X.shape[1]
    hidden_dim = 256
    output_dim = PREDICTION_HORIZON  # 24 de iesiri
    num_layers = 3

    model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Antrenarea modelului
    epochs = 200
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        #Evaluarea pe setul de validare
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

    print(f"\nCladire: {building_id} | Loss Final - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

    # Evaluare pe setul de test
    model.eval()
    y_test_pred = []
    y_test_actual = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            y_test_pred.append(y_pred.numpy())
            y_test_actual.append(y_batch.numpy())


    y_test_pred = np.concatenate(y_test_pred, axis=0)
    y_test_actual = np.concatenate(y_test_actual, axis=0)

    y_test_pred = scaler_y.inverse_transform(y_test_pred)
    y_test_actual = scaler_y.inverse_transform(y_test_actual)

    # Calcul metrici
    mse = mean_squared_error(y_test_actual, y_test_pred, multioutput='uniform_average')
    mae = mean_absolute_error(y_test_actual, y_test_pred, multioutput='uniform_average')
    r2 = r2_score(y_test_actual, y_test_pred, multioutput='uniform_average')
    smape = np.mean(2 * np.abs(y_test_pred - y_test_actual) / (np.abs(y_test_pred) + np.abs(y_test_actual))) * 100

    print(f" Cladire: {building_id}, MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}, SMAPE: {smape:.2f}%")

    #Salvez metricile in log
    metrics_log.append([building_id, mse, mae, r2, smape])

    # Ajustez dimensiunile pentru a fi egale
    min_length = min(len(timestamps), len(y_test_pred.flatten()))
    timestamps = timestamps[:min_length]
    y_test_actual = y_test_actual.flatten()[:min_length]
    y_test_pred = y_test_pred.flatten()[:min_length]

    # Construiesc DataFrame-ul corect
    result = pd.DataFrame({
        'timestamp': timestamps,
        'actual': y_test_actual,
        'predicted': y_test_pred,
        'error': y_test_actual - y_test_pred
    })

    result = result.drop_duplicates(subset=['timestamp']).sort_values(by='timestamp')

    #Salvez in CSV
    building_folder = os.path.join(output_folder, f'building_{building_id}')
    os.makedirs(building_folder, exist_ok=True)
    result.to_csv(os.path.join(building_folder, f'LSTM_24h_{building_id}.csv'), index=False)


    # Grafic rezultate
    plt.figure(figsize=(12, 6))
    plt.plot(result['timestamp'][:250], result['actual'][:250], label='Valori Reale', color='blue')
    plt.plot(result['timestamp'][:250], result['predicted'][:250], label='Valori Prezise', color='red')
    plt.xlabel('Zi')
    plt.ylabel('Consum de energie')
    plt.title(f'Predictie zilnica LSTM pentru {building_id}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(os.path.join(building_folder, f'LSTM_daily_graph_{building_id}.png'))
    plt.close()

#Salvez metricile intr-un fisier CSV
metrics_df = pd.DataFrame(metrics_log, columns=['Building', 'MSE', 'MAE', 'R2', 'SMAPE'])
metrics_df.to_csv('lstm_metrics.csv', index=False)

print("\nToate predictiile pentru cele 30 de cladiri au fost finalizate!")

