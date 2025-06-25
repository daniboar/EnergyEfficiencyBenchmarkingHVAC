import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

# === CONFIGURARE ===
BUILDING_ID = 'Panther_office_Larry'
DATE = '2017-12-14_Thursday'

BASELINE_PATH = f'../baseline/baselines_days_output/{BUILDING_ID}/Tuesday/{BUILDING_ID}_Tuesday_baseline.csv'
PREDICTION_PATH = f'../prediction_for_a_day/prediction_for_2017-12-14_Thursday/{BUILDING_ID}/prediction_{BUILDING_ID}_{DATE}.csv'
OUTPUT_FOLDER = 'mpc_optimization_output'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === PARAMETRI OPTIMIZARE ===
GAMMA = 0.5     # penalizare pentru peak
Z_MIN = 0.5     # minim admis pentru z(t)
Z_MAX = 0.9
H = 24          # orizont de optimizare: 24h
PEAK_HOURS = list(range(9, 18))

# === INCARCARE DATE ===
baseline_df = pd.read_csv(BASELINE_PATH)
pred_df = pd.read_csv(PREDICTION_PATH)

# Medie pe ore pentru baseline
baseline_today = baseline_df.groupby("hour")["B(t)"].mean().reindex(range(24)).ffill().values
prediction = pred_df['predicted_consumption'].values[:24]
B_avg = (baseline_today + prediction) / 2

# === FUNCTIE OBIECTIV ===
def objective(z, B, gamma):
    total_consumption = np.sum(B * z)
    peak_demand = np.max(B * z)
    return total_consumption + gamma * peak_demand

# === CONSTRANGERI ===
def peak_constraints(z):
    # z(t) == 1 pentru orele de vârf
    return [z[t] - 1.0 for t in PEAK_HOURS] + [1.0 - z[t] for t in PEAK_HOURS]

# === SETARI PENTRU OPTIMIZARE ===
bounds = [(Z_MIN, Z_MAX) if t not in PEAK_HOURS else (1.0, 1.0) for t in range(H)]
z0 = np.ones(H)
constraints = {'type': 'ineq', 'fun': lambda z: np.array(peak_constraints(z))}

# === OPTIMIZARE ===
res = minimize(objective, z0, args=(B_avg, GAMMA), method='trust-constr', bounds=bounds, constraints=constraints)

# === EXTRAGERE SI SALVARE ===
z_opt = res.x
optimized_consumption = B_avg * z_opt

# DataFrame cu rezultate
result_df = pd.DataFrame({
    'hour': range(24),
    'baseline': baseline_today,
    'prediction': prediction,
    'B_avg': B_avg,
    'z(t)': z_opt,
    'optimized_consumption': optimized_consumption
})

# Salvare CSV si grafic
csv_path = os.path.join(OUTPUT_FOLDER, f'mpc_optimized_{BUILDING_ID}.csv')
plot_path = os.path.join(OUTPUT_FOLDER, f'mpc_optimized_{BUILDING_ID}.png')
result_df.to_csv(csv_path, index=False)

# === GRAFIC ===
plt.figure(figsize=(12, 6))
plt.plot(result_df['hour'], result_df['baseline'], '--', label='Baseline')
plt.plot(result_df['hour'], result_df['prediction'], '--', label='Prediction')
plt.plot(result_df['hour'], result_df['optimized_consumption'], linewidth=2, label='Optimized Consumption')
plt.fill_between(result_df['hour'], result_df['optimized_consumption'], alpha=0.3)
plt.title(f'MPC Optimization - {BUILDING_ID} - {DATE}')
plt.xlabel('Hour')
plt.ylabel('Energy Consumption (kWh)')
plt.xticks(range(0, 24))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(plot_path)
plt.close()

print(f"Optimizare completă. Fișier CSV salvat la: {csv_path}")
