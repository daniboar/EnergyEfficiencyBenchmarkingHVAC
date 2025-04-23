import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === CONFIG ===
energy_path = '../electricity_30_kWh.csv'
weather_path = '../weather_Panther.csv'
output_folder = 'building_heatmaps'
os.makedirs(output_folder, exist_ok=True)

# === Citire date ===
energy_df = pd.read_csv(energy_path)
weather_df = pd.read_csv(weather_path)
energy_df['timestamp'] = pd.to_datetime(energy_df['timestamp'])
weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])

# Selectam doar coloanele meteo relevante
weather_df = weather_df[['timestamp', 'airTemperature', 'dewTemperature', 'cloudCoverage',
                         'precipDepth1HR', 'precipDepth6HR', 'seaLvlPressure',
                         'windDirection', 'windSpeed']]

# === Cladiri ===
building_columns = [col for col in energy_df.columns if col != 'timestamp']
summary_corrs = {}

# === Corelatie si heatmap per cladire ===
for building_id in building_columns:
    temp_df = energy_df[['timestamp', building_id]].dropna()
    merged_df = pd.merge(temp_df, weather_df, on='timestamp')
    merged_df.dropna(inplace=True)

    if merged_df.empty:
        continue

    corr_matrix = merged_df.drop(columns=['timestamp']).corr(method='pearson')
    corr_series = corr_matrix[building_id].drop(building_id)
    summary_corrs[building_id] = corr_series

    # Heatmap individual
    plt.figure(figsize=(9, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title(f'Heatmap Corelatii - {building_id}')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'heatmap_{building_id}.png'))
    plt.close()

# === Heatmap general: medii absolute ===
summary_df = pd.DataFrame(summary_corrs).T.abs()
mean_corrs = summary_df.mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=mean_corrs.index, y=mean_corrs.values, palette='viridis')
plt.title('Corelatie medie absoluta intre caracteristici meteo si consumul de energie (toate cladirile)')
plt.ylabel('Valoare medie')
plt.xlabel('Caracteristica meteo')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('heatmap_general_weather_features.png')
plt.show()
