import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Incarc datele din csv, si aleg primele 30 de cladiri + timestamp-ul
data = pd.read_csv('electricity_cleaned.csv')
data = data.iloc[:, :31]
output_folder = 'mlp_predictions'
os.makedirs(output_folder, exist_ok=True)


# 2. Functie pentru generarea caracteristicilor temporale (sliding window)
def create_time_series_features(df, target_column, window_size=3):
    df = df.copy()
    for lag in range(1, window_size + 1):
        df[f'lag_{lag}'] = df[target_column].shift(lag)

    # Caracteristici temporale
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)  # Weekend: sâmbata/duminica

    # Sezon (1 = iarna, 2 = primăvara, 3 = vara, 4 = toamna)
    df['season'] = df['month'].apply(
        lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
    )

    # One-hot encoding pentru ziua saptamanii si sezon
    df = pd.get_dummies(df, columns=['weekday', 'season'])

    df.dropna(inplace=True)
    return df


# 3. Definesc modelului MLP optimizat
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x.view(-1)


# 4. Definesc datasetul pentru PyTorch
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 3. Iterez prin cele 30 de cladiri pentru a face predictia si salvarea in CSV
building_columns = data.columns[1:31]

cnt = 0
for building_id in building_columns:
    cnt += 1
    print(f"\nIncepe procesarea pentru cladirea {cnt}: {building_id}")
    building_data = data[['timestamp', building_id]].dropna()

    # Transform timestamp intr-un format datetime
    building_data['timestamp'] = pd.to_datetime(building_data['timestamp'])
    building_data.set_index('timestamp', inplace=True)

    # Caracteristici temporale de baza
    building_data = create_time_series_features(building_data, building_id, window_size=3)

    # Separăm X și y
    X = building_data.drop(columns=[building_id])
    y = building_data[building_id]

    # Normalizare (StandardScaler)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Impart setul de date in train, test si validare (80% pentru antrenare si 10% pentru testare si 10% pentru validare)
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Construiesc modelul MLP
    input_dim = X.shape[1]
    hidden_dim = 128
    model = MLPModel(input_dim, hidden_dim)

    # Folosesc optimizatorul Adam cu un learning rate de 0.0005
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # Antrenez modelul (200 epoci)
    epochs = 200
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch, y_batch.view(-1)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

    # Evaluez modelul
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            predictions.append(y_pred.numpy())
            actuals.append(y_batch.numpy())

    y_test_pred = np.concatenate(predictions)
    y_test_actual = np.concatenate(actuals)

    # Calculez metricile
    mse = mean_squared_error(y_test_actual, y_test_pred)
    mae = mean_absolute_error(y_test_actual, y_test_pred)
    r2 = r2_score(y_test_actual, y_test_pred)
    smape = np.mean(2 * np.abs(y_test_pred - y_test_actual) / (np.abs(y_test_pred) + np.abs(y_test_actual))) * 100

    print(f"Cladire: {building_id}, MSE: {mse:.2f}, MAE: {mae:.2f}, R^2: {r2:.2f}, SMAPE: {smape:.2f}%")

    # Salvăm rezultatele
    result = pd.DataFrame({'timestamp': building_data.index[-len(y_test_actual):],
                           'actual': y_test_actual,
                           'predicted': y_test_pred,
                           'MSE': mse,
                           'MAE': mae,
                           'R2': r2,
                           'SMAPE': smape})

    building_folder = os.path.join(output_folder, f'building_{building_id}')
    os.makedirs(building_folder, exist_ok=True)

    result.to_csv(os.path.join(building_folder, f'MLP_results_{building_id}.csv'), index=False)

    # Grafic
    plt.figure(figsize=(12, 6))
    plt.plot(result['timestamp'][:250], result['actual'][:250], label='Valori Reale', color='blue')
    plt.plot(result['timestamp'][:250], result['predicted'][:250], label='Valori Prezise', color='red')
    plt.legend()
    plt.savefig(os.path.join(building_folder, f'MLP_graph_{building_id}.png'))
    plt.close()

print("\nToate predictiile pentru cele 30 de cladiri au fost finalizate!")
