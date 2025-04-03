import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 1. Incarc datele din csv, si aleg primele 30 de cladiri + timestamp-ul
data = pd.read_csv('electricity_cleaned_kWh.csv')
data = data.iloc[:, :31]  # Timestamp + primele 30 de cladiri
output_folder = 'random_forest_daily_predictions'
os.makedirs(output_folder, exist_ok=True)

metrics_log = []

# 2. Functie pentru agregarea datelor pe zile + adaugarea caracteristicilor avansate
def aggregate_daily(df, target_column):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Agregam prin medie
    df_daily = df.resample('D').mean()

    # Lag features (valori anterioare)
    for lag in range(1, 4):  # 3 zile anterioare
        df_daily[f'lag_{lag}'] = df_daily[target_column].shift(lag)

    # Caracteristici temporale
    df_daily['day'] = df_daily.index.day
    df_daily['month'] = df_daily.index.month
    df_daily['day_of_year'] = df_daily.index.dayofyear  # Ziua din an (1-365)
    df_daily['week_of_year'] = df_daily.index.isocalendar().week  # Saptamana anului (1-52)
    df_daily['weekday'] = df_daily.index.weekday
    df_daily['is_weekend'] = df_daily['weekday'].apply(lambda x: 1 if x >= 5 else 0)

    # Anotimpuri (1 = iarna, 2 = primavara, 3 = vara, 4 = toamna)
    df_daily['season'] = df_daily['month'].apply(
        lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
    )

    # One-hot encoding pentru ziua saptamanii si sezon
    df_daily = pd.get_dummies(df_daily, columns=['weekday', 'season'])

    df_daily.dropna(inplace=True)
    return df_daily


# 3. Iterez prin cele 30 de cladiri
building_columns = data.columns[1:31]

cnt = 0
for building_id in building_columns:
    cnt += 1
    print(f"\nProcesez cladirea {cnt}: {building_id}")
    building_data = data[['timestamp', building_id]].dropna()

    # Prelucrez datele la nivel de zi + adaug caracteristici avansate
    building_data = aggregate_daily(building_data, building_id)

    # Separ caracteristicile (X) și tinta (y)
    X = building_data.drop(columns=[building_id])
    y = building_data[building_id]

    # Impart datele in train (80%), test (10%) si validare (10%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

    # Construiesc modelul Random Forest
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Evaluez modelul pe setul de validare
    y_val_pred = model.predict(X_val)

    # Evaluez modelul pe setul de test
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_smape = np.mean(2 * np.abs(y_test_pred - y_test) / (np.abs(y_test_pred) + np.abs(y_test))) * 100

    print(f"Cladire: {building_id}, MSE: {test_mse:.2f}, MAE: {test_mae:.2f}, R²: {test_r2:.2f}, SMAPE: {test_smape:.2f}%")

    #Salvez metricile in log
    metrics_log.append([building_id, test_mse, test_mae, test_r2, test_smape])

    # Realizez predictii pentru intreaga serie
    all_predictions = model.predict(X)

    # Salvez rezultatele intr-un DataFrame
    result = pd.DataFrame({'timestamp': building_data.index,
                           'actual': y,
                           'predicted': all_predictions,
                           'error': y - all_predictions})

    # Salvez rezultatele intr-un folder separat pentru fiecare cladire
    building_folder = os.path.join(output_folder, f'building_{building_id}')
    os.makedirs(building_folder, exist_ok=True)

    csv_output_path = os.path.join(building_folder, f'predicted_{building_id}.csv')
    result.to_csv(csv_output_path, index=False)

    # Generez un grafic comparativ
    plt.figure(figsize=(12, 6))
    plt.plot(result['timestamp'][:250], result['actual'][:250], label='Valori Reale', color='blue')
    plt.plot(result['timestamp'][:250], result['predicted'][:250], label='Valori Prezise', color='red')
    plt.xlabel('Data')
    plt.ylabel('Consum de energie')
    plt.title(f'Predicție Daily Random Forest pentru {building_id}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    graph_output_path = os.path.join(building_folder, f'graph_{building_id}.png')
    plt.savefig(graph_output_path)
    plt.close()

#Salvez metricile intr-un fisier CSV
metrics_df = pd.DataFrame(metrics_log, columns=['Building', 'MSE', 'MAE', 'R2', 'SMAPE'])
metrics_df.to_csv('random_forest_metrics.csv', index=False)

print("\nToate predictiile pentru cele 30 de cladiri au fost finalizate!")