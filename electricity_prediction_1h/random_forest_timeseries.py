import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 1. Incarc datele din csv, si aleg primele 30 de cladiri + timestamp-ul
data = pd.read_csv('electricity_cleaned.csv')
data = data.iloc[:, :31]
output_folder = 'random_forest_timeseries_predictions'
os.makedirs(output_folder, exist_ok=True)

# 2. Funcție pentru generarea caracteristicilor temporale (sliding window)
def create_time_series_features(df, target_column, window_size=3):
    df = df.copy()
    for lag in range(1, window_size + 1):
        df[f'lag_{lag}'] = df[target_column].shift(lag)

    # Adaug caracteristici temporale suplimentare
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)  # Weekend: sambata/duminica

    # Anotimpuri: 1=iarna, 2=primavara, 3=vara, 4=toamna
    df['season'] = df['month'].apply(
        lambda x: 1 if x in [12, 1, 2] else 2 if x in [3, 4, 5] else 3 if x in [6, 7, 8] else 4
    )

    # Convertesc 'weekday' in valori numerice folosind one-hot encoding
    df = pd.get_dummies(df, columns=['weekday', 'season'])

    df.dropna(inplace=True)
    return df

# 3. Iterez prin cele 30 de cladiri pentru a face predictia si salvarea in CSV
building_columns = data.columns[1:31]

cnt = 0
for building_id in building_columns:
    cnt += 1
    print(f"\nProcesez cladirea {cnt}: {building_id}")
    building_data = data[['timestamp', building_id]].dropna()

    # Transform timestamp intr-un format datetime
    building_data['timestamp'] = pd.to_datetime(building_data['timestamp'])
    building_data.set_index('timestamp', inplace=True)

    # Aplic generarea caracteristicilor temporale
    building_data = create_time_series_features(building_data, building_id, window_size=3)

    # Separăm caracteristicile (X) și ținta (y)
    X = building_data.drop(columns=[building_id])
    y = building_data[building_id]

    # Impart setul de date in train, test si validare (80% pentru antrenare si 10% pentru testare si 10% pentru validare)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

    # Construim modelul Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluez modelul pe setul de validare
    y_val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    print(f"Mean Squared Error (MSE) pe setul de validare: {val_mse:.2f}")

    # Evaluez modelul pe setul de test
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_smape = np.mean(2 * np.abs(y_test_pred - y_test) / (np.abs(y_test_pred) + np.abs(y_test))) * 100

    print(f"Test Metrics pentru {building_id}:")
    print(f"  - MSE: {test_mse:.2f}")
    print(f"  - MAE: {test_mae:.2f}")
    print(f"  - R²: {test_r2:.2f}")
    print(f"  - SMAPE: {test_smape:.2f}%")

    # Realizez predicții pentru intreaga serie
    all_predictions = model.predict(X)

    # Salvez rezultatele într-un DataFrame
    result = pd.DataFrame({'timestamp': building_data.index,
                           'actual': y,
                           'predicted': all_predictions,
                           'error': y - all_predictions})

    # Adaug metricile in CSV
    result['MSE'] = test_mse
    result['MAE'] = test_mae
    result['R2'] = test_r2
    result['SMAPE'] = test_smape

    building_folder = os.path.join(output_folder, f'building_{building_id}')
    os.makedirs(building_folder, exist_ok=True)

    csv_output_path = os.path.join(building_folder, f'predicted_{building_id}.csv')
    result.to_csv(csv_output_path, index=False)

    # 8. Generez un grafic comparativ
    plt.figure(figsize=(12, 6))
    plt.plot(result['timestamp'][:250], result['actual'][:250], label='Valori Reale', color='blue')
    plt.plot(result['timestamp'][:250], result['predicted'][:250], label='Valori Prezise', color='red')
    plt.xlabel('Timestamp')
    plt.ylabel('Consum de energie')
    plt.title(f'Predicție Time Series Random Forest pentru {building_id}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    graph_output_path = os.path.join(building_folder, f'graph_{building_id}.png')
    plt.savefig(graph_output_path)
    plt.close()

print("\nToate predictiile pentru cele 30 de cladiri au fost finalizate!")
