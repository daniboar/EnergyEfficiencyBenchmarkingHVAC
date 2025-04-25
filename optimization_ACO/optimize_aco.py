import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# === DETECTARE peek/offpeek ===
def detect_peek_hours(baseline):
    mean_val = np.mean(baseline)
    std_val = np.std(baseline)
    threshold = mean_val + 0.5 * std_val
    peek_hours = [t for t, b in enumerate(baseline) if b > threshold]
    offpeek_hours = [t for t in range(24) if t not in peek_hours]
    return peek_hours, offpeek_hours


# === FITNESS FUNCTION ===
def fitness_aco(schedule, prediction, baseline, peek_hours, offpeek_hours):
    penalty = 0
    for t in range(24):
        adjusted = prediction[t] * schedule[t]
        deviation = abs(adjusted - baseline[t])
        if t in peek_hours:
            weight = 2
        elif t in offpeek_hours:
            weight = 0.5
        else:
            weight = 1
        penalty += deviation * weight
    for i in range(1, 24):
        if schedule[i] != schedule[i - 1]:
            penalty += 0.5  # penalizare switching
    return penalty

# === ACO OPTIMIZATION ===
def run_aco(prediction, baseline, peek_hours, offpeek_hours, n_ants=40, n_iterations=100, alpha=1, beta=2,
            evaporation=0.2):
    pheromone = np.ones((24, 2))  # 0 = off, 1 = on
    best_schedule = None
    best_score = float('inf')

    for _ in range(n_iterations):
        schedules = []
        scores = []

        for _ in range(n_ants):
            schedule = []
            for t in range(24):
                tau = pheromone[t]
                eta = [1.0, 1.0]
                prob_0 = (tau[0] ** alpha) * (eta[0] ** beta)
                prob_1 = (tau[1] ** alpha) * (eta[1] ** beta)
                total = prob_0 + prob_1
                prob = [prob_0 / total, prob_1 / total]
                choice = np.random.choice([0, 1], p=prob)
                schedule.append(choice)
            schedules.append(schedule)
            score = fitness_aco(schedule, prediction, baseline, peek_hours, offpeek_hours)
            scores.append(score)

            if score < best_score:
                best_score = score
                best_schedule = schedule

        # actualizare feromoni
        pheromone *= (1 - evaporation)
        for schedule, score in zip(schedules, scores):
            for t in range(24):
                pheromone[t][schedule[t]] += 1.0 / (score + 1e-6)

    return best_schedule


# === FUNCTIA PRINCIPALA ===
def optimize_consum_aco(building_id: str, target_date: str):
    day_name = pd.to_datetime(target_date).day_name()
    profile_dir = f"../profil_de_consum/profil_consum_{target_date}_{building_id}"
    profile_csv = os.path.join(profile_dir, f"profil_consum_{building_id}_{target_date}.csv")

    if not os.path.exists(profile_csv):
        raise FileNotFoundError(f"Fisierul CSV nu exista: {profile_csv}")

    df = pd.read_csv(profile_csv)
    prediction = df["predicted_P(t)"].values
    baseline = df["baseline_B(t)"].values

    peek_hours, offpeek_hours = detect_peek_hours(baseline)
    print("Peak hours: ", peek_hours)
    print("Offpeek hours: ", offpeek_hours)

    optimal_schedule = run_aco(prediction, baseline, peek_hours, offpeek_hours)

    df["HVAC_ACO"] = optimal_schedule
    df["P(t)_adjusted_ACO"] = prediction * df["HVAC_ACO"]

    # === SALVARE ===
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, f"{building_id}_{target_date}")
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, f"profil_OPTIMIZAT_ACO_{building_id}_{target_date}.csv")
    plot_path = os.path.join(output_dir, f"graf_OPTIMIZARE_ACO_{building_id}_{target_date}.png")

    df.to_csv(output_csv, index=False)

    # === PLOT ===
    plt.figure(figsize=(14, 6))
    plt.plot(df['hour'], df['baseline_B(t)'], label='Baseline B(t)', color='blue', marker='o')
    plt.plot(df['hour'], df['predicted_P(t)'], label='Predictie P(t)', color='green', marker='o')
    plt.plot(df['hour'], df['P(t)_adjusted_ACO'], label='Consum optimizat ACO', color='orange', marker='o')
    plt.title(f'Optimizare ACO HVAC - {building_id} - {target_date} ({day_name})')
    plt.xlabel('Ora (0-23)')
    plt.ylabel('Consum energie [kWh]')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xticks(np.arange(0, 24, 1))
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"CSV salvat in: {output_csv}")
    print(f"Grafic salvat in: {plot_path}")


# === ENTRY POINT ===
if __name__ == "__main__":
    optimize_consum_aco(
        building_id="Panther_office_Catherine",
        target_date="2017-12-14"
    )
