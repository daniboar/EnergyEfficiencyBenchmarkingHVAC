import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# DETECTARE peek/offpeek din baseline
def detect_peek_hours(baseline):
    mean_val = np.mean(baseline)
    std_val = np.std(baseline)
    threshold = mean_val + 0.5 * std_val
    peek_hours = [t for t, b in enumerate(baseline) if b > threshold]
    offpeek_hours = [t for t in range(24) if t not in peek_hours]
    return peek_hours, offpeek_hours


# FUNCTIA DE FITNESS CU ponderi dinamice
def fitness_pso(position, prediction, baseline, peek_hours, offpeek_hours):
    penalty = 0
    for t in range(24):
        adjusted = prediction[t] * position[t]
        deviation = abs(adjusted - baseline[t])

        if t in peek_hours:
            weight = 2
        elif t in offpeek_hours:
            weight = 0.5
        else:
            weight = 1

        penalty += deviation * weight
    return penalty


# IMPLEMENTARE PSO
def run_pso(prediction, baseline, peek_hours, offpeek_hours, n_particles=30, iterations=100):
    levels = np.array([0.25, 0.5, 0.75, 1.0])
    positions = np.random.choice(levels, size=(n_particles, 24))
    velocities = np.random.randn(n_particles, 24) * 0.1
    personal_best_positions = positions.copy()
    personal_best_scores = np.array(
        [fitness_pso(pos, prediction, baseline, peek_hours, offpeek_hours) for pos in positions])
    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]

    w, c1, c2 = 0.5, 1.5, 1.5

    for _ in range(iterations):
        for i in range(n_particles):
            r1, r2 = np.random.rand(2)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (personal_best_positions[i] - positions[i]) +
                             c2 * r2 * (global_best_position - positions[i]))
            positions[i] += velocities[i]

            # rotunjesc la cel mai apropiat nivel
            positions[i] = np.array([levels[np.argmin(abs(levels - p))] for p in positions[i]])

            score = fitness_pso(positions[i], prediction, baseline, peek_hours, offpeek_hours)
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = positions[i]

        best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[best_idx]

    return global_best_position


# FUNCTIE PRINCIPALA
def optimize_consum_pso(building_id: str, target_date: str):
    day_name = pd.to_datetime(target_date).day_name()
    profile_dir = f"../profil_de_consum/profil_consum_{target_date}_{building_id}"
    profile_csv = os.path.join(profile_dir, f"profil_consum_{building_id}_{target_date}.csv")

    if not os.path.exists(profile_csv):
        raise FileNotFoundError(f"Fisierul CSV nu exista: {profile_csv}")

    df = pd.read_csv(profile_csv)
    prediction = df["predicted_P(t)"].values
    baseline = df["baseline_B(t)"].values

    # detectez peek/offpeak automat
    peek_hours, offpeek_hours = detect_peek_hours(baseline)

    print("Peak hours: ", peek_hours)
    print("Offpeek hours: ", offpeek_hours)

    # optimizare
    optimal_intensity = run_pso(prediction, baseline, peek_hours, offpeek_hours)
    df["HVAC_PSO"] = optimal_intensity
    df["P(t)_adjusted_PSO"] = prediction * optimal_intensity

    # salvare output
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, f"{building_id}_{target_date}")
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, f"profil_OPTIMIZAT_PSO_{building_id}_{target_date}.csv")
    plot_path = os.path.join(output_dir, f"graf_OPTIMIZARE_PSO_{building_id}_{target_date}.png")

    df.to_csv(output_csv, index=False)

    # plot
    plt.figure(figsize=(14, 6))
    plt.plot(df['hour'], df['baseline_B(t)'], label='Baseline B(t)', color='blue', marker='o')
    plt.plot(df['hour'], df['predicted_P(t)'], label='Predictie P(t)', color='green', marker='o')
    plt.plot(df['hour'], df['P(t)_adjusted_PSO'], label='Consum optimizat PSO', color='orange', marker='o')
    plt.title(f'Optimizare PSO HVAC - {building_id} - {target_date} ({day_name})')
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


# MAIN
if __name__ == "__main__":
    optimize_consum_pso(
        building_id="Panther_education_Misty",
        target_date="2017-12-30"
    )
