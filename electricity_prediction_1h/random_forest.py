import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Incarc datele din csv, si aleg primele 30 de cladiri + timestamp-ul
data = pd.read_csv('electricity_cleaned.csv')
data = data.iloc[:, :31]
output_folder = 'random_forest_predictions'
os.makedirs(output_folder, exist_ok=True)

# 2. Iterez prin cele 30 de cladiri pentru a face precizia si a salva intr-un csv aceasta predictie
building_columns = data.columns[1:31]

cnt = 0
for building_id in building_columns:
    cnt = cnt + 1
    print(f"\nIncepe procesarea pentru cladirea {cnt}: {building_id}")
    building_data = data[['timestamp', building_id]].dropna()

    # Transform timestamp intr-un format datetime
    building_data['timestamp'] = pd.to_datetime(building_data['timestamp'])
    building_data.set_index('timestamp', inplace=True)

    # Extrag informatii esentiale pentru ora, zi, luna si numele zilei din saptamana
    building_data['hour'] = building_data.index.hour
    building_data['day'] = building_data.index.day
    building_data['month'] = building_data.index.month
    building_data['weekday'] = building_data.index.day_name()

    # Convertesc 'weekday' in valori numerice folosind one-hot encoding
    building_data = pd.get_dummies(building_data, columns=['weekday'])
    # print(building_data)

    # Definesc X (caracteristici din coloana timestamp) și y (consumul de energie pentru cladirea selectata)
    X = building_data.drop(columns=[building_id])
    y = building_data[building_id]

    # Impart setul de date in train, test si validare (80% pentru antrenare si 10% pentru testare si 10% pentru validare)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Construiesc modelul Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluez modelul pe setul de validare
    y_val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    print(f"Mean Squared Error (MSE) pe setul de validare: {val_mse:.2f}")

    # Evaluez modelul pe setul de test
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    print(f"Mean Squared Error (MSE) pe setul de test: {test_mse:.2f}")

    # Realizez predictii pentru toata perioada (01.01.2016 - 31.12.2017)
    all_predictions = model.predict(X)

    result = pd.DataFrame({'timestamp': building_data.index,
                           'actual': y,
                           'predicted': all_predictions})

    # Salvez intr-un csv acest informatii (timestamp, valoarea actuala, valoarea prezisa)
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
    plt.title(f'Predicție Random Forest pentru {building_id}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    graph_output_path = os.path.join(building_folder, f'graph_{building_id}.png')
    plt.savefig(graph_output_path)
    plt.close()

print("\nToate predictiile au fost finalizate!")
