import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Incarc datele din csv, si aleg primele 30 de cladiri + timestamp-ul
data = pd.read_csv('electricity_cleaned.csv')
data = data.iloc[:, :31]
output_folder = 'lstm_predictions'
os.makedirs(output_folder, exist_ok=True)


# 2. Definesc modelul LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


# Definesc datasetul pentru Torch
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Functie pentru generarea caracteristicilor temporale (sliding window)
def create_time_series_features(df, target_column, window_size=3):
    df = df.copy()
    for lag in range(1, window_size + 1):
        df[f'lag_{lag}'] = df[target_column].shift(lag)

    # Caracteristici temporale
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)  # Weekend: sambata/duminica

    # Sezon (1 = iarna, 2 = primavara, 3 = vara, 4 = toamna)
    df['season'] = df['month'].apply(
        lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
    )

    # One-hot encoding pentru ziua saptamanii si sezon
    df = pd.get_dummies(df, columns=['weekday', 'season'])

    df.dropna(inplace=True)
    return df


# 3. Iterez prin cele 30 de cladiri pentru a face predictia si salvarea in CSV
building_columns = data.columns[1:31]

cnt = 0
for building_id in building_columns:
    cnt = cnt + 1
    print(f"\nIncepe procesarea pentru cladirea {cnt}: {building_id}")
    building_data = data[['timestamp', building_id]].dropna()

    # Transform timestamp intr-un format datetime
    building_data['timestamp'] = pd.to_datetime(building_data['timestamp'])
    building_data.set_index('timestamp', inplace=True)

    # Caracteristici temporale de baza
    building_data = create_time_series_features(building_data, building_id, window_size=3)

    # Separ X și y
    X = building_data.drop(columns=[building_id])
    y = building_data[building_id]

    # Normalizez datele de intrare
    X_normalized = (X - X.mean()) / X.std()

    # Impart setul de date in train, test si validare (80% pentru antrenare si 10% pentru testare si 10% pentru validare)
    X_train, X_temp, y_train, y_temp = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Construiesc modelul
    input_dim = X.shape[1]
    hidden_dim = 128
    output_dim = 1
    num_layers = 2

    model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # Antrenarea modelului
    epochs = 100
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validare
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()
        val_loss /= len(val_loader)

    # Evaluarea pe setul de test
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            predictions.append(y_pred.numpy())
            actuals.append(y_batch.numpy())
    y_test_pred = np.concatenate(predictions)
    y_test_actual = np.concatenate(actuals)

    # Calculez metricile de performanta
    mse = mean_squared_error(y_test_actual, y_test_pred)
    mae = mean_absolute_error(y_test_actual, y_test_pred)
    r2 = r2_score(y_test_actual, y_test_pred)
    smape = np.mean(2 * np.abs(y_test_pred - y_test_actual) / (np.abs(y_test_pred) + np.abs(y_test_actual))) * 100

    print(f"Cladire: {building_id}, MSE: {mse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}, SMAPE: {smape:.2f}%")

    # Salvez rezultatele intr-un CSV
    result = pd.DataFrame({'timestamp': building_data.index[-len(y_test_actual):],
                           'actual': y_test_actual.flatten(),
                           'predicted': y_test_pred.flatten()})
    result['error'] = result['actual'] - result['predicted']
    result['MSE'] = mse
    result['MAE'] = mae
    result['R2'] = r2
    result['SMAPE'] = smape

    building_folder = os.path.join(output_folder, f'building_{building_id}')
    os.makedirs(building_folder, exist_ok=True)

    csv_output_path = os.path.join(building_folder, f'LSTM_results_{building_id}.csv')
    result.to_csv(csv_output_path, index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(result['timestamp'][:250], result['actual'][:250], label='Valori Reale', color='blue')
    plt.plot(result['timestamp'][:250], result['predicted'][:250], label='Valori Prezise', color='red')
    plt.xlabel('Timestamp')
    plt.ylabel('Consum de energie')
    plt.title(f'Comparatie Real vs Prezis pentru {building_id}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    graph_path = os.path.join(building_folder, f'LSTM_graph_{building_id}.png')
    plt.savefig(graph_path)
    plt.close()

print("\nToate predictiile pentru cele 30 de cladiri au fost finalizate!")
