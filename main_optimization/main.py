import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from optimization_ACO.optimize_aco import optimize_consum_aco
from optimization_GA.optimize_ga import optimize_consum_ga
from optimization_PSO.optimize_pso import optimize_consum_pso
from profil_de_consum.profil import generate_energy_profile


def compare_optimizations(building_id: str, target_date: str):
    """
    Genereaza un plot comparativ de optimizare folosind 3 tipuri de optimizatori pentru o cladire si o zi data.

    Args:
        building_id (str): Numele cladirii
        target_date (str): Data in format 'YYYY-MM-DD'
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    generate_energy_profile(building_id, target_date)
    optimize_consum_aco(building_id, target_date)
    optimize_consum_ga(building_id, target_date)
    optimize_consum_pso(building_id, target_date)

    path_pso = f"../optimization_PSO/{building_id}_{target_date}/profil_OPTIMIZAT_PSO_{building_id}_{target_date}.csv"
    path_ga = f"../optimization_GA/{building_id}_{target_date}/profil_OPTIMIZAT_GA_{building_id}_{target_date}.csv"
    path_aco = f"../optimization_ACO/{building_id}_{target_date}/profil_OPTIMIZAT_ACO_{building_id}_{target_date}.csv"

    # Incarc fisierele CSV
    df_ga = pd.read_csv(path_ga)
    df_pso = pd.read_csv(path_pso)
    df_aco = pd.read_csv(path_aco)

    date_day = pd.Timestamp(target_date)

    # Grafic comparativ
    plt.figure(figsize=(14, 6))
    plt.plot(df_ga['hour'], df_ga['real_R(t)'], label='Real R(t)', color='blue', linestyle='--')
    plt.plot(df_ga['hour'], df_ga['baseline_B(t)'], label='Baseline B(t)', color='green', linestyle='--')
    plt.plot(df_ga['hour'], df_ga['predicted_P(t)'], label='Predictie P(t)', color='red', linestyle='--')

    plt.plot(df_ga['hour'], df_ga['P(t)_adjusted_GA'], label='GA', color='black', marker='o')
    plt.plot(df_pso['hour'], df_pso['P(t)_adjusted_PSO'], label='PSO', color='orange', marker='s')
    plt.plot(df_aco['hour'], df_aco['P(t)_adjusted_ACO'], label='ACO', color='purple', marker='^')

    plt.title(f'Comparatie Optimizari HVAC - {building_id} - {target_date} - {date_day.day_name()}')
    plt.xlabel('Ora (0-23)')
    plt.ylabel('Consum energie [kWh]')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xticks(np.arange(0, 24, 1))
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Salvare in acelasi folder cu acest script
    plot_output_dir = os.path.join(base_dir, f"{building_id}_{target_date}_{date_day.day_name()}")
    os.makedirs(plot_output_dir, exist_ok=True)

    plot_path = os.path.join(plot_output_dir, f"comparatie_{building_id}_{target_date}_{date_day.day_name()}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Grafic comparativ salvat in: {plot_path}")

    # Salvare CSV comparativ
    df_final = pd.DataFrame({
        "hour": df_ga["hour"],
        "real_R(t)": df_ga["real_R(t)"],
        "predicted_P(t)": df_ga["predicted_P(t)"],
        "baseline_B(t)": df_ga["baseline_B(t)"],
        "P(t)_adjusted_GA": df_ga["P(t)_adjusted_GA"],
        "P(t)_adjusted_PSO": df_pso["P(t)_adjusted_PSO"],
        "P(t)_adjusted_ACO": df_aco["P(t)_adjusted_ACO"],
    })

    csv_path = os.path.join(plot_output_dir, f"comparatie_{building_id}_{target_date}_{date_day.day_name()}.csv")
    df_final.to_csv(csv_path, index=False)
    print(f"CSV comparativ salvat in: {csv_path}")

if __name__ == "__main__":
    compare_optimizations(
        building_id="Panther_office_Catherine",
        target_date="2017-12-12"
    )
