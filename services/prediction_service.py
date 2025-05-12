import os
from collections import OrderedDict

import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from config import DB_URI
from models.baseline_model import Baseline
from models.real_consumption_model import RealConsumption
from models.prediction_model import Prediction, Base
from profil_de_consum.profil import generate_energy_profile

engine = create_engine(DB_URI)
Session = sessionmaker(bind=engine)
Base.metadata.create_all(engine)


def predict_for_day(building_name, target_date):
    generate_energy_profile(building_name, target_date)
    csv_path = f'../profil_de_consum/profil_consum_{target_date}_{building_name}/profil_consum_{building_name}_{target_date}.csv'

    if not os.path.exists(csv_path):
        return {"error": "Fisierul CSV cu predictii nu a fost generat"}, 404

    df = pd.read_csv(csv_path)
    predicted = df['predicted_P(t)'].tolist()
    real = df['real_R(t)'].tolist()
    baseline = df['baseline_B(t)'].tolist()

    with Session() as session:
        # Prediction insert/update
        existing_pred = session.get(Prediction, (building_name, target_date))
        if existing_pred:
            for i in range(24):
                setattr(existing_pred, f'h{i}', predicted[i])
        else:
            prediction = Prediction(
                building_name=building_name,
                target_date=target_date,
                **{f"h{i}": predicted[i] for i in range(24)}
            )
            session.add(prediction)

        # RealConsumption insert/update
        existing_real = session.get(RealConsumption, (building_name, target_date))
        if existing_real:
            for i in range(24):
                setattr(existing_real, f'h{i}', real[i])
        else:
            real_consumption = RealConsumption(
                building_name=building_name,
                target_date=target_date,
                **{f"h{i}": real[i] for i in range(24)}
            )
            session.add(real_consumption)

        # Baseline insert/update
        existing_real = session.get(Baseline, (building_name, target_date))
        if existing_real:
            for i in range(24):
                setattr(existing_real, f'h{i}', baseline[i])
        else:
            baseline = Baseline(
                building_name=building_name,
                target_date=target_date,
                **{f"h{i}": baseline[i] for i in range(24)}
            )
            session.add(baseline)

        session.commit()

    return {
        "status": f"Predictia, consumul real si baseline-ul au fost salvate pentru {building_name} in {target_date}"}, 200


def get_prediction_for_day(building_name, target_date):
    with Session() as session:
        result = session.get(Prediction, (building_name, target_date))
        if not result:
            return {"error": "Nu exista predictie"}, 404
        return {
            "building_name": result.building_name,
            "target_date": str(result.target_date),
            "predicted": [{"hour": i, "value": getattr(result, f"h{i}")} for i in range(24)]
        }, 200


def get_real_consumption_for_day(building_name, target_date):
    with Session() as session:
        result = session.get(RealConsumption, (building_name, target_date))
        if not result:
            return {"error": "Nu exista consum real"}, 404

        return {
            "building_name": result.building_name,
            "target_date": str(result.target_date),
            "real_consumption": [{"hour": i, "value": getattr(result, f"h{i}")} for i in range(24)]
        }, 200


def get_baseline_for_day(building_name, target_date):
    with Session() as session:
        result = session.get(Baseline, (building_name, target_date))
        if not result:
            return {"error": "Nu exista baseline"}, 404
        return {
            "building_name": result.building_name,
            "target_date": str(result.target_date),
            "baseline": [{"hour": i, "value": getattr(result, f"h{i}")} for i in range(24)]
        }, 200


def get_combined_profile(building_name, target_date):
    with Session() as session:
        pred = session.get(Prediction, (building_name, target_date))
        real = session.get(RealConsumption, (building_name, target_date))
        baseline = session.get(Baseline, (building_name, target_date))

        if not pred or not real or not baseline:
            return {"error": "Nu exista date"}, 404

        values = []
        for i in range(24):
            values.append({
                "hour": i,
                "consumption": {
                    "prediction": getattr(pred, f"h{i}") if pred else None,
                    "real": getattr(real, f"h{i}") if real else None,
                    "baseline": getattr(baseline, f"h{i}") if baseline else None
                }
            })

        return {
            "building_name": building_name,
            "target_date": target_date,
            "values": values
        }, 200
