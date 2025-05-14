import os

import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from config import DB_URI
from main_optimization.main import compare_optimizations
from models.aco_model import AcoOptimization
from models.baseline_model import Baseline
from models.dew_temperature_model import DewTemperature
from models.ga_model import GaOptimization
from models.pso_model import PsoOptimization
from models.real_consumption_model import RealConsumption
from models.prediction_model import Prediction, Base
from models.air_temperature_model import AirTemperature

engine = create_engine(DB_URI)
Session = sessionmaker(bind=engine)
Base.metadata.create_all(engine)


def predict_for_day(building_name, target_date):
    compare_optimizations(building_name, target_date)
    date_day = pd.Timestamp(target_date)
    csv_path = f'../profil_de_consum/profil_consum_{target_date}_{building_name}/profil_consum_{building_name}_{target_date}.csv'
    csv_path_optimization = f'../main_optimization/{building_name}_{target_date}_{date_day.day_name()}/comparatie_{building_name}_{target_date}_{date_day.day_name()}.csv'
    if not os.path.exists(csv_path):
        return {"error": "Fisierul CSV cu predictii nu a fost generat"}, 404

    df = pd.read_csv(csv_path)
    df_optimization = pd.read_csv(csv_path_optimization)

    predicted = df['predicted_P(t)'].tolist()
    real = df['real_R(t)'].tolist()
    baseline = df['baseline_B(t)'].tolist()
    ga_optimization = df_optimization['P(t)_adjusted_GA'].tolist()
    pso_optimization = df_optimization['P(t)_adjusted_PSO'].tolist()
    aco_optimization = df_optimization['P(t)_adjusted_ACO'].tolist()
    mse = df['mse'].tolist()
    mae = df['mae'].tolist()
    smape = df['smape'].tolist()
    r2 = df['r2'].tolist()
    airTemperature = df['airTemperature'].tolist()
    dewTemperature = df['dewTemperature'].tolist()

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
                **{f"h{i}": predicted[i] for i in range(24)},
                mse=mse[0],
                mae=mae[0],
                smape=smape[0],
                r2=r2[0]
            )
            session.add(prediction)

        # AirTemperature insert/update
        existing_air_temp = session.get(AirTemperature, target_date)
        if existing_air_temp:
            for i in range(24):
                setattr(existing_air_temp, f'h{i}', airTemperature[i])
        else:
            air_temp = AirTemperature(
                target_date=target_date,
                **{f"h{i}": airTemperature[i] for i in range(24)}
            )
            session.add(air_temp)

        # DewTemperature insert/update
        existing_dew_temp = session.get(DewTemperature, target_date)
        if existing_dew_temp:
            for i in range(24):
                setattr(existing_dew_temp, f'h{i}', dewTemperature[i])
        else:
            dew_temp = DewTemperature(
                target_date=target_date,
                **{f"h{i}": dewTemperature[i] for i in range(24)}
            )
            session.add(dew_temp)

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

        # Ga optimization insert/update
        existing_ga = session.get(GaOptimization, (building_name, target_date))
        if existing_ga:
            for i in range(24):
                setattr(existing_ga, f'h{i}', ga_optimization[i])
        else:
            ga_values = GaOptimization(
                building_name=building_name,
                target_date=target_date,
                **{f"h{i}": ga_optimization[i] for i in range(24)}
            )
            session.add(ga_values)

        session.commit()

        # Pso optimization insert/update
        existing_pso = session.get(PsoOptimization, (building_name, target_date))
        if existing_pso:
            for i in range(24):
                setattr(existing_pso, f'h{i}', pso_optimization[i])
        else:
            pso_values = PsoOptimization(
                building_name=building_name,
                target_date=target_date,
                **{f"h{i}": pso_optimization[i] for i in range(24)}
            )
            session.add(pso_values)

        session.commit()

        # Aso optimization insert/update
        existing_aco = session.get(AcoOptimization, (building_name, target_date))
        if existing_aco:
            for i in range(24):
                setattr(existing_aco, f'h{i}', aco_optimization[i])
        else:
            aco_values = AcoOptimization(
                building_name=building_name,
                target_date=target_date,
                **{f"h{i}": aco_optimization[i] for i in range(24)}
            )
            session.add(aco_values)

        session.commit()
    return {
        "status": f"Datele au fost salvate {building_name} in {target_date}"}, 200


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
        airTemperature = session.get(AirTemperature, target_date)
        dewTemperature = session.get(DewTemperature, target_date)
        ga = session.get(GaOptimization, (building_name, target_date))
        pso = session.get(PsoOptimization, (building_name, target_date))
        aco = session.get(AcoOptimization, (building_name, target_date))

        if not pred or not real or not baseline or not airTemperature or not dewTemperature:
            return {"error": "Nu exista date"}, 404

        values = []
        for i in range(24):
            values.append({
                "hour": i,
                "consumption": {
                    "prediction": getattr(pred, f"h{i}") if pred else None,
                    "real": getattr(real, f"h{i}") if real else None,
                    "baseline": getattr(baseline, f"h{i}") if baseline else None,
                    "gaOptimization": getattr(ga, f"h{i}") if ga else None,
                    "psoOptimization": getattr(pso, f"h{i}") if pso else None,
                    "acoOptimization": getattr(aco, f"h{i}") if aco else None
                },
                "airTemperature": getattr(airTemperature, f"h{i}") if airTemperature else None,
                "dewTemperature": getattr(dewTemperature, f"h{i}") if dewTemperature else None,
            })

        return {
            "building_name": building_name,
            "target_date": target_date,
            "values": values,
            "mse": pred.mse,
            "mae": pred.mae,
            "smape": pred.smape,
            "r2": pred.r2,
        }, 200
