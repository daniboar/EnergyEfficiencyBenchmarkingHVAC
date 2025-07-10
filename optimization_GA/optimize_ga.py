import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# NIVELE POSIBILE HVAC
levels = [0.25, 0.5, 0.75, 1.0]


# DETECTARE ORE DE VARF
def detect_peek_hours(baseline):
    mean_val = np.mean(baseline)
    std_val = np.std(baseline)
    threshold = mean_val + 0.5 * std_val
    peek_hours = [t for t, b in enumerate(baseline) if b > threshold]
    offpeek_hours = [t for t in range(24) if t not in peek_hours]
    return peek_hours, offpeek_hours


# FITNESS FUNCTION CU peek/offpeak
def fitness(individual, prediction, baseline, peek_hours, offpeek_hours):
    total_penalty = 0
    for hour in range(24):
        hvac_action = individual[hour]
        adjusted = prediction[hour] * hvac_action
        deviation = abs(adjusted - baseline[hour])

        if hour in peek_hours:
            weight = 2
        elif hour in offpeek_hours:
            weight = 0.5
        else:
            weight = 1

        total_penalty += deviation * weight

    for i in range(1, 24):
        if individual[i] != individual[i - 1]:
            total_penalty += 0.5  # penalizare pentru switching

    return -total_penalty


#  COMPONENTE GA
def initialize_population(pop_size):
    return [[random.choice(levels) for _ in range(24)] for _ in range(pop_size)]


def selection(population, fitnesses):
    return random.choices(population, weights=fitnesses, k=2)


def crossover(p1, p2):
    cut = random.randint(1, 22)
    return p1[:cut] + p2[cut:], p2[:cut] + p1[cut:]


def mutate(individual, rate=0.1):
    return [
        random.choice(levels) if random.random() < rate else gene
        for gene in individual
    ]


def run_ga(prediction, baseline, peek_hours, offpeek_hours, generations=100, pop_size=50):
    population = initialize_population(pop_size)
    for _ in range(generations):
        raw_scores = [fitness(ind, prediction, baseline, peek_hours, offpeek_hours) for ind in population]
        min_score = min(raw_scores)
        fitnesses = [score - min_score + 1e-6 for score in raw_scores]  # toate pozitive

        new_population = []
        for _ in range(pop_size // 2):
            p1, p2 = selection(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            new_population.extend([mutate(c1), mutate(c2)])
        population = new_population

    best = max(population, key=lambda ind: fitness(ind, prediction, baseline, peek_hours, offpeek_hours))
    return best


# OPTIMIZARE PRINCIPALA
def optimize_consum_ga(building_id: str, target_date: str):
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
    print("Offpeak hours: ", offpeek_hours)

    optimal_schedule = run_ga(prediction, baseline, peek_hours, offpeek_hours)
    df["HVAC_GA"] = optimal_schedule
    df["P(t)_adjusted_GA"] = prediction * df["HVAC_GA"]

    # SALVARE
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, f"{building_id}_{target_date}")
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, f"profil_OPTIMIZAT_GA_{building_id}_{target_date}.csv")
    plot_path = os.path.join(output_dir, f"graf_OPTIMIZARE_GA_{building_id}_{target_date}.png")

    df.to_csv(output_csv, index=False)

    # PLOT
    plt.figure(figsize=(14, 6))
    plt.plot(df['hour'], df['baseline_B(t)'], label='Baseline B(t)', color='blue', marker='o')
    plt.plot(df['hour'], df['predicted_P(t)'], label='Predictie P(t)', color='green', marker='o')
    plt.plot(df['hour'], df['P(t)_adjusted_GA'], label='Consum optimizat GA', color='orange', marker='o')
    plt.title(f'Optimizare GA HVAC - {building_id} - {target_date} ({day_name})')
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
    optimize_consum_ga(
        building_id="Panther_education_Misty",
        target_date="2017-12-30"
    )
