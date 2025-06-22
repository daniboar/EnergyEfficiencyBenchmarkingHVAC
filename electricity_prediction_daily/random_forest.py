import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Incarc datele din csv
data = pd.read_csv('electricity_cleaned_kWh.csv')
data = data.iloc[:, :31]  # Timestamp + primele 30 de clădiri
output_folder = 'random_forest_daily_predictions'
os.makedirs(output_folder, exist_ok=True)

# Parametrii modelului
LOOKBACK_DAYS = 3  # Folosesc ultimele 3 zile (3 * 24 ore) pentru predicție
PREDICTION_HORIZON = 24  # Vreau sa prezic pentru urmatoarele 24 de ore

metrics_log = []

# 2. Functie pentru generarea caracteristicilor temporale
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
        lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
    )

    # One-hot encoding pentru ziua saptamanii si sezon
    df = pd.get_dummies(df, columns=['weekday', 'season'])

    # Creez lag-uri pentru ultimele `lookback_days` zile (fiecare zi are 24 ore)
    for lag in range(1, lookback_days * 24 + 1):
        df[f'lag_{lag}'] = df[target_column].shift(lag)

    df.dropna(inplace=True)
    return df


# 3. Iterez prin cele 30 de cladiri și antrenez modelele
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

        # Salvezm toate cele 24 de timestamp-uri asociate predicției
        timestamps.append(building_data.index[i + 1: i + 1 + PREDICTION_HORIZON].tolist())

    # Adaug timestamp-urile intr-o singura lista (flatten)
    timestamps = [ts for sublist in timestamps for ts in sublist]

    X = np.array(X)
    y = np.array(y)

    # Împart datele în train (80%), validare (10%), test (10%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

    # Construiesc modelul Random Forest
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Fac predicții pe setul de test
    y_test_pred = model.predict(X_test)

    # Calculez metricile de performanta (MSE, MAE, R², SMAPE)
    mse = mean_squared_error(y_test, y_test_pred, multioutput='uniform_average')
    mae = mean_absolute_error(y_test, y_test_pred, multioutput='uniform_average')
    r2 = r2_score(y_test, y_test_pred, multioutput='uniform_average')
    smape = np.mean(2 * np.abs(y_test_pred - y_test) / (np.abs(y_test_pred) + np.abs(y_test))) * 100

    print(f"Rezultate pentru {building_id}: MSE={mse:.2f}, MAE={mae:.2f}, R²={r2:.2f}, SMAPE={smape:.2f}%")

    # Salvez metricile în log
    metrics_log.append([building_id, mse, mae, r2, smape])

    # Ajustez dimensiunile pentru a fi egale
    min_length = min(len(timestamps), len(y_test_pred.flatten()))
    timestamps = timestamps[:min_length]
    y_test_actual = y_test.flatten()[:min_length]
    y_test_pred = y_test_pred.flatten()[:min_length]

    # Construiesc DataFrame-ul corect
    result = pd.DataFrame({
        'timestamp': timestamps,
        'actual': y_test_actual,
        'predicted': y_test_pred,
        'error': y_test_actual - y_test_pred
    })

    result = result.drop_duplicates(subset=['timestamp']).sort_values(by='timestamp')

    # Salvez în CSV
    building_folder = os.path.join(output_folder, f'building_{building_id}')
    os.makedirs(building_folder, exist_ok=True)
    result.to_csv(os.path.join(building_folder, f'RandomForest_24h_{building_id}.csv'), index=False)

    # Generez un grafic comparativ
    plt.figure(figsize=(12, 6))
    plt.plot(result['timestamp'][:250], result['actual'][:250], label='Valori Reale', color='blue')
    plt.plot(result['timestamp'][:250], result['predicted'][:250], label='Valori Prezise', color='red')
    plt.xlabel('Data')
    plt.ylabel('Consum de energie')
    plt.title(f'Predictie Daily Random Forest pentru {building_id}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    graph_output_path = os.path.join(building_folder, f'graph_{building_id}.png')
    plt.savefig(graph_output_path)
    plt.close()

# Salvez metricile intr-un fisier CSV
metrics_df = pd.DataFrame(metrics_log, columns=['Building', 'MSE', 'MAE', 'R2', 'SMAPE'])
metrics_df.to_csv('random_forest_metrics.csv', index=False)

print("\nToate predictiile pentru cele 30 de cladiri au fost finalizate!")