import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from prediction_for_a_day.prediction_for_a_day_LSTM import predict_energy_for_day


# Generez profilul de consum pentru o cladire si o zi data.
def generate_energy_profile(building_id: str, target_date: str):
    # PARSEZ DATA SI ZIUA SAPTAMANII
    date_obj = pd.to_datetime(target_date)
    day_name = date_obj.day_name()  # ex: Monday, Tuesday, Wednesday...

    # APELEZ FUNCTIA DE PREDICTIE
    predict_energy_for_day(building_id, target_date)

    # CONFIGUREZ CAILE
    baseline_path = f'../baseline/baselines_days_output/{building_id}/{day_name}/{building_id}_{day_name}_baseline.csv'
    prediction_path = f'../prediction_for_a_day/prediction_for_{target_date}_{day_name}/{building_id}/prediction_{building_id}_{target_date}_{day_name}.csv'
    base_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(base_dir, f'profil_consum_{target_date}_{building_id}')
    os.makedirs(output_dir, exist_ok=True)

    # INCARC DATE
    if not os.path.exists(baseline_path):
        print(f"Fisierul baseline lipseste: {baseline_path}")
        return
    if not os.path.exists(prediction_path):
        print(f"Fisierul predictie lipseste: {prediction_path}")
        return

    baseline = pd.read_csv(baseline_path)
    predictie = pd.read_csv(prediction_path)

    # COMBINARE DATE
    profil = pd.DataFrame({
        'hour': baseline['hour'],
        'baseline_B(t)': baseline['B(t)'],
        'predicted_P(t)': predictie['predicted_consumption'],
        'real_R(t)': predictie.get('actual_consumption', np.nan),
        'mse': predictie['mse'],
        'mae': predictie['mae'],
        'smape': predictie['smape'],
        'r2': predictie['r2'],
        'airTemperature': predictie['airTemperature'],
        'dewTemperature': predictie['dewTemperature'],
    })

    # INCARC DATELE METEO
    weather_path = os.path.join(base_dir, '../weather_Panther.csv')
    if os.path.exists(weather_path):
        weather_df = pd.read_csv(weather_path, parse_dates=['timestamp'])

        # Extrag ziua respectiva
        mask = weather_df['timestamp'].dt.date == date_obj.date()
        weather_day = weather_df.loc[mask]

        # Creez o coloana "hour" pentru join
        weather_day = weather_day.copy()  # te asiguri ca e o copie sigura
        weather_day['hour'] = weather_day['timestamp'].dt.hour

        # Selectez doar coloanele dorite
        weather_cols = weather_day[['hour', 'windSpeed', 'windDirection', 'precipDepth1HR']]

        # Fac merge cu profilul (pe ora)
        profil = pd.merge(profil, weather_cols, on='hour', how='left')
        profil[['windSpeed', 'windDirection', 'precipDepth1HR']] = profil[
            ['windSpeed', 'windDirection', 'precipDepth1HR']].fillna(0)
    else:
        print(f"Fisierul de vreme lipseste: {weather_path}")

    profil['P(t)_deviation_%'] = 100 * (profil['predicted_P(t)'] - profil['baseline_B(t)']) / profil['baseline_B(t)']
    profil['R(t)_deviation_%'] = 100 * (profil['real_R(t)'] - profil['baseline_B(t)']) / profil['baseline_B(t)']

    # SALVARE CSV
    profil_csv_path = os.path.join(output_dir, f'profil_consum_{building_id}_{target_date}.csv')
    profil.to_csv(profil_csv_path, index=False)
    print(f"Profilul a fost salvat in {profil_csv_path}")

    # GRAFIC 1: Profil Real vs Predictie vs Baseline
    plt.figure(figsize=(14, 6))
    plt.plot(profil['hour'], profil['baseline_B(t)'], label='Baseline B(t)', color='blue', marker='o')
    plt.plot(profil['hour'], profil['predicted_P(t)'], label='Predictie P(t)', color='green', marker='o')
    if profil['real_R(t)'].notna().all():
        plt.plot(profil['hour'], profil['real_R(t)'], label='Real R(t)', color='red', marker='o')

    plt.title(f'Profil de consum - {building_id} - {target_date} ({day_name})', fontsize=16)
    plt.xlabel('Ora (0-23)', fontsize=14)
    plt.ylabel('Consum [kWh]', fontsize=14)
    plt.xticks(np.arange(0, 24, 1), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'graf_consum_{building_id}_{target_date}.png'))
    plt.close()

    # GRAFIC 2: Abatere fata de Baseline
    plt.figure(figsize=(14, 6))
    plt.plot(profil['hour'], profil['P(t)_deviation_%'], label='P(t) vs B(t)', color='green', marker='o')
    if profil['real_R(t)'].notna().all():
        plt.plot(profil['hour'], profil['R(t)_deviation_%'], label='R(t) vs B(t)', color='red', marker='o')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    plt.title(f'Abatere procentuala fata de baseline - {building_id} - {target_date}', fontsize=16)
    plt.xlabel('Ora', fontsize=14)
    plt.ylabel('Deviatie (%)', fontsize=14)
    plt.xticks(np.arange(0, 24, 1), fontsize=12)
    plt.yticks(np.arange(0, 100, 10), fontsize=12)
    plt.grid(axis='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'graf_deviatie_{building_id}_{target_date}.png'))
    plt.close()

    print(f"Graficele au fost salvate in folderul {output_dir}")


# MAIN
if __name__ == '__main__':
    generate_energy_profile('Panther_office_Catherine', '2017-12-15')
