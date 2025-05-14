import pandas as pd
import numpy as np
import os

ROLLING_DAYS = 7  # numarul de zile anterioare
csv_file = '../electricity_30_kWh.csv'
output_folder = 'baselines_day'
os.makedirs(output_folder, exist_ok=True)

def calculate_baseline_for_date(building_id, target_date_str):
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    target_date = pd.to_datetime(target_date_str)
    day_idx = target_date.weekday()
    day_name = target_date.day_name()

    building_data = df[[building_id]].dropna()
    building_data['weekday'] = building_data.index.weekday
    building_data['hour'] = building_data.index.hour

    day_data = building_data[building_data['weekday'] == day_idx]
    past_days = sorted(day_data.index.normalize().unique())
    past_days = [d for d in past_days if d < target_date]

    past_days = past_days[-ROLLING_DAYS:]
    past_data = []

    for day in past_days:
        daily = day_data[day_data.index.normalize() == day][building_id].values
        if len(daily) == 24:
            past_data.append(daily)

    if len(past_data) == 0:
        print(f"[!] Nu sunt suficiente date pentru {building_id} - {day_name} anterior datei {target_date_str}")
        return

    past_data = np.array(past_data)
    y_mean = past_data.mean(axis=0)

    # Creez path: baselines_day/<building_id>/<target_date>/
    day_folder = os.path.join(output_folder, building_id, target_date_str)
    os.makedirs(day_folder, exist_ok=True)

    baseline_df = pd.DataFrame({'hour': range(24), 'B(t)': y_mean})
    csv_path = os.path.join(day_folder, f'baseline_{building_id}_{target_date_str}.csv')
    baseline_df.to_csv(csv_path, index=False)

    print(f"[✔] Baseline salvat in {csv_path}")

# === MAIN ===
if __name__ == '__main__':
    # exemplu de test
    calculate_baseline_for_date("Panther_office_Catherine", "2017-12-14")
