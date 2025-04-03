import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

csv_file = '../electricity_30_kWh.csv'
output_folder = 'baselines_dataset'
os.makedirs(output_folder, exist_ok=True)

SEASONS = {
    'winter': [12, 1, 2],
    'spring': [3, 4, 5],
    'summer': [6, 7, 8],
    'autumn': [9, 10, 11]
}

MONTHS_FOR_SEASON = {
    'winter': 1,
    'spring': 4,
    'summer': 7,
    'autumn': 10
}

SEMESTERS = {
    'S1': [1, 2, 3, 4, 5, 6],
    'S2': [7, 8, 9, 10, 11, 12]
}

def compute_baseline(building_data, mask):
    filtered = building_data.loc[mask]
    baseline = filtered.groupby(filtered.index.hour).mean().reset_index()
    baseline.columns = ['hour', 'B(t)']
    return baseline

def plot_baseline(baseline, building_id, mask_name, folder_path):
    x = baseline['hour']
    y = baseline['B(t)']

    x_smooth = np.linspace(x.min(), x.max(), 300)
    spline = make_interp_spline(x, y)
    y_smooth = spline(x_smooth)

    plt.figure(figsize=(12, 6))
    plt.plot(x_smooth, y_smooth, color='blue', label=f'Baseline {mask_name}')
    plt.scatter(x, y, color='red', label='Valori orare')
    plt.xlabel('Ora')
    plt.ylabel('Consum mediu energie')
    plt.title(f'Profil orar consum energie - {building_id} ({mask_name})')
    plt.grid(True)
    plt.legend()
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, f'{building_id}_baseline_{mask_name}.png'))
    plt.close()

def process_building(building_data, building_id):
    building_folder = os.path.join(output_folder, f'{building_id}')
    os.makedirs(building_folder, exist_ok=True)

    baselines = []

    # Monthly Baseline
    monthly_folder = os.path.join(building_folder, 'monthly')
    os.makedirs(monthly_folder, exist_ok=True)
    for season, month in MONTHS_FOR_SEASON.items():
        mask = building_data.index.month == month
        baseline = compute_baseline(building_data, mask)
        baseline.to_csv(os.path.join(monthly_folder, f'{building_id}_baseline_{season}_month.csv'), index=False)
        plot_baseline(baseline, building_id, f'{season}_month', monthly_folder)
        baselines.append(baseline['B(t)'])

    # Seasonal Baseline
    seasonal_folder = os.path.join(building_folder, 'seasonal')
    os.makedirs(seasonal_folder, exist_ok=True)
    for season, months in SEASONS.items():
        mask = building_data.index.month.isin(months)
        baseline = compute_baseline(building_data, mask)
        baseline.to_csv(os.path.join(seasonal_folder, f'{building_id}_baseline_{season}.csv'), index=False)
        plot_baseline(baseline, building_id, f'{season}', seasonal_folder)
        baselines.append(baseline['B(t)'])

    # Semestrial Baseline
    semestrial_folder = os.path.join(building_folder, 'semestrial')
    os.makedirs(semestrial_folder, exist_ok=True)
    for semester, months in SEMESTERS.items():
        mask = building_data.index.month.isin(months)
        baseline = compute_baseline(building_data, mask)
        baseline.to_csv(os.path.join(semestrial_folder, f'{building_id}_baseline_{semester}.csv'), index=False)
        plot_baseline(baseline, building_id, f'{semester}', semestrial_folder)
        baselines.append(baseline['B(t)'])

    # Final Baseline
    final_baseline = pd.concat(baselines, axis=1).mean(axis=1)
    result = pd.DataFrame({
        'hour': range(24),
        'B(t)_final': final_baseline
    })
    result.to_csv(os.path.join(building_folder, f'{building_id}_baseline_FINAL.csv'), index=False)

    # Plot final baseline
    plt.figure(figsize=(12, 6))
    plt.plot(result['hour'], result['B(t)_final'], color='green', label='Baseline Final')
    plt.scatter(result['hour'], result['B(t)_final'], color='orange', label='Valori medii finale')
    plt.xlabel('Ora')
    plt.ylabel('Consum mediu energie')
    plt.title(f'Baseline Final - {building_id}')
    plt.grid(True)
    plt.legend()
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(os.path.join(building_folder, f'{building_id}_baseline_FINAL.png'))
    plt.close()

def main():
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    cnt = 0
    for building_id in df.columns:
        cnt += 1
        print(f" Procesez cladirea {cnt}: {building_id}")
        building_data = df[[building_id]].dropna()
        process_building(building_data, building_id)

    print("\n Toate baseline-urile au fost calculate si salvate!")

main()
