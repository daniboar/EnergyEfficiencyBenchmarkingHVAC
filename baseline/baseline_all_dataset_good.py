import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

ROLLING_DAYS = 7
INPUT_FILE = '../electricity_30_kWh.csv'
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, 'baselines_dataset_building')
os.makedirs(output_dir, exist_ok=True)


def generate_full_baseline_for_building(building_id):
    df = pd.read_csv(INPUT_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    if building_id not in df.columns:
        print(f"Cladirea {building_id} nu exista in dataset.")
        return

    building_data = df[[building_id]].dropna()
    building_data['weekday'] = building_data.index.weekday

    all_days = sorted(building_data.index.normalize().unique())

    full_result = []

    for target_date in all_days:
        day_idx = target_date.weekday()
        day_data = building_data[building_data['weekday'] == day_idx]

        # Cauta ultimele 7 zile inainte de target_date
        previous_days = [d for d in day_data.index.normalize().unique() if d < target_date]
        previous_days = previous_days[-ROLLING_DAYS:]

        past_data = []
        for d in previous_days:
            try:
                vals = building_data.loc[pd.date_range(d, periods=24, freq='h')][building_id].values
                if len(vals) == 24:
                    past_data.append(vals)
            except:
                continue

        if len(past_data) < ROLLING_DAYS:
            continue

        baseline_avg = np.array(past_data).mean(axis=0)

        # extrag consumul real
        try:
            timestamps = pd.date_range(target_date, periods=24, freq='h')
            real_vals = building_data.loc[timestamps][building_id].values
            if len(real_vals) != 24:
                continue
        except:
            continue

        for h in range(24):
            full_result.append({
                "timestamp": timestamps[h],
                "baseline": baseline_avg[h],
                "real": real_vals[h]
            })

    # === salvare CSV ===
    if not full_result:
        print("Nu s-a generat niciun rezultat.")
        return

    result_df = pd.DataFrame(full_result)
    folder = os.path.join(output_dir, building_id)
    os.makedirs(folder, exist_ok=True)

    csv_path = os.path.join(folder, f"{building_id}_baseline_vs_real.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"CSV salvat in: {csv_path}")

    # === plot ===
    plt.figure(figsize=(18, 6))
    plt.plot(result_df['timestamp'], result_df['real'], label='Real', color='blue', linewidth=1)
    plt.plot(result_df['timestamp'], result_df['baseline'], label='Baseline', color='orange', linestyle='--')
    plt.title(f'Consum real vs baseline - {building_id}')
    plt.xlabel('Timp')
    plt.ylabel('Consum (kWh)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(folder, f"{building_id}_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot salvat in: {plot_path}")


# === MAIN ===
if __name__ == '__main__':
    generate_full_baseline_for_building("Panther_education_Misty")
