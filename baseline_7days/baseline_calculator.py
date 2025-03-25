import pandas as pd
import os

ROLLING_DAYS = 7  # numarul de zile pentru baseline
csv_file = '../electricity_30.csv'
output_folder = 'baselines_output'

# Creăm doar folderul principal 'baselines_output'
os.makedirs(output_folder, exist_ok=True)


def calculate_hourly_baseline(file_name):
    print(f"\n Procesez cladirile...")

    df = pd.read_csv(file_name)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Parcurg fiecare cladire
    cnt = 0
    for building_id in df.columns:
        cnt += 1
        print(f" → Cladire {cnt}: {building_id}")
        building_data = df[[building_id]].dropna()

        # Creez o masca pentru a selecta doar datele din ultimele ROLLING_DAYS
        mask = building_data.index >= (building_data.index.max() - pd.Timedelta(days=ROLLING_DAYS))

        # Filtrez doar datele din ultimele ROLLING_DAYS
        building_data_filtered = building_data.loc[mask]

        # Calculez media orara pe ultimele ROLLING_DAYS
        baseline = building_data_filtered.groupby(building_data_filtered.index.hour)[building_id].mean().reset_index()
        baseline.columns = ['hour', 'B(t)']

        output_path = os.path.join(output_folder, f'{building_id}_baseline.csv')
        baseline.to_csv(output_path, index=False)


calculate_hourly_baseline(csv_file)

print("\n Calculul baseline-urilor pentru electricity a fost finalizat si salvat!")
