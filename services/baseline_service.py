import os

import pandas as pd
from flask import jsonify

from baseline.baseline_all_dataset_good import generate_full_baseline_for_building


def baseline_dataset(building_id):
    generate_full_baseline_for_building(building_id)
    print(building_id)
    csv_path = f"../baseline/baselines_dataset_building/{building_id}/{building_id}_baseline_vs_real.csv"
    print(csv_path)
    if not os.path.exists(csv_path):
        return {"error": "Fisierul nu exista"}, 404

    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    data = []
    for _, row in df.iterrows():
        data.append({
            "timestamp": row['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
            "baseline": round(row['baseline'], 2),
            "real": round(row['real'], 2)
        })

    return jsonify({
        "building_name": building_id,
        "data": data
    })
