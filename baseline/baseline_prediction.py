import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

ROLLING_DAYS = 7  # numarul de zile anterioare
csv_file = '../electricity_30.csv'
output_folder = 'baselines_days_output'
os.makedirs(output_folder, exist_ok=True)

day_map = {
    0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
    4: 'Friday', 5: 'Saturday', 6: 'Sunday'
}

def calculate_baseline_per_day(file_name):
    df = pd.read_csv(file_name)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    cnt = 0
    for building_id in df.columns:
        cnt += 1
        print(f"\n Procesez cladirea {cnt}: {building_id}")
        building_folder = os.path.join(output_folder, building_id)
        os.makedirs(building_folder, exist_ok=True)

        building_data = df[[building_id]].dropna()
        building_data['weekday'] = building_data.index.weekday
        building_data['hour'] = building_data.index.hour

        for day_idx, day_name in day_map.items():
            print(f" → Ziua: {day_name}")
            day_folder = os.path.join(building_folder, day_name)
            os.makedirs(day_folder, exist_ok=True)

            # Selectez toate datele pentru ziua respectivă
            day_data = building_data[building_data['weekday'] == day_idx]

            # Aleg ultimele ROLLING_DAYS zile disponibile
            unique_days = day_data.index.normalize().unique()[-ROLLING_DAYS:]
            past_data = []

            for day in unique_days:
                daily = day_data[day_data.index.normalize() == day][building_id].values
                if len(daily) == 24:
                    past_data.append(daily)

            if len(past_data) == 0:
                print(f"Nu sunt suficiente zile pentru {day_name}")
                continue

            past_data = np.array(past_data)

            # Media pe fiecare ora
            y_mean = past_data.mean(axis=0)

            # Salvez baseline în CSV
            baseline_df = pd.DataFrame({
                'hour': range(24),
                'B(t)': y_mean
            })
            baseline_df.to_csv(os.path.join(day_folder, f'{building_id}_{day_name}_baseline.csv'), index=False)

            # Generez plot
            plt.figure(figsize=(10, 5))
            plt.plot(range(24), y_mean, color='blue', label='Baseline')
            plt.scatter(range(24), y_mean, color='red', label='Valori medii finale')
            plt.xlabel('Ora')
            plt.ylabel('Consum mediu energie')
            plt.title(f'Baseline consum energie ({day_name}) - {building_id}')
            plt.grid(True)
            plt.xticks(range(24))
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(day_folder, f'{building_id}_{day_name}_baseline_plot.png'))
            plt.close()

    print("\nCalculul baseline-urilor pe zile a fost finalizat.")

calculate_baseline_per_day(csv_file)
