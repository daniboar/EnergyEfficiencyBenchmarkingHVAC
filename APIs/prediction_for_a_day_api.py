from flask import Flask, jsonify
from sqlalchemy import create_engine, text
import os
import pandas as pd
from flasgger import Swagger, swag_from
from prediction_for_a_day.prediction_for_a_day_LSTM import predict_energy_for_day

app = Flask(__name__)
swagger = Swagger(app)

DB_URI = os.getenv('DB_URI', 'postgresql://postgres:postgres@localhost:5432/licenta')
engine = create_engine(DB_URI)


def populate_buildings(building_list):
    with engine.begin() as conn:
        for name in building_list:
            conn.execute(text("INSERT INTO buildings (name) VALUES (:name) ON CONFLICT DO NOTHING"), {"name": name})


@swag_from({
    'tags': ['prediction'],
    'description': 'Endpoint that calculate prediction for a day and actual consumption',
    'parameters': [
        {
            'name': 'building_name',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'Numele cladirii pentru care se face predictia'
        },
        {
            'name': 'target_date',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'Data pentru predictie (format YYYY-MM-DD)'
        }
    ],
    'responses': {
        200: {
            'description': 'Predictie generata si salvata in baza de date pentru cladirea specificata'
        },
        404: {
            'description': 'CSV-ul de predictie lipseste dupa rulare'
        }
    }
})
@app.route('/predict/<string:building_name>/<string:target_date>', methods=['POST'])
def predict_one(building_name, target_date):
    predict_energy_for_day(building_name, target_date)
    day_name = pd.Timestamp(target_date).day_name()
    csv_path = f'../prediction_for_a_day/prediction_for_{target_date}_{day_name}/{building_name}/prediction_{building_name}_{target_date}_{day_name}.csv'

    if not os.path.exists(csv_path):
        return jsonify({"error": "Fisierul CSV cu predictii nu a fost generat"}), 404

    df = pd.read_csv(csv_path)
    predicted_values = df['predicted_consumption'].tolist()
    real_values = df['actual_consumption'].tolist()

    with engine.begin() as conn:
        # === INSERT INTO predictions ===
        conn.execute(
            text(f"""
                INSERT INTO predictions (building_name, target_date, {','.join([f'h{i}' for i in range(24)])})
                VALUES (:building_name, :target_date, {','.join([f':h{i}' for i in range(24)])})
                ON CONFLICT (building_name, target_date) DO UPDATE SET
                {', '.join([f'h{i} = EXCLUDED.h{i}' for i in range(24)])}
            """),
            {
                "building_name": building_name,
                "target_date": target_date,
                **{f"h{i}": predicted_values[i] for i in range(24)}
            }
        )

        # === INSERT INTO realconsumption ===
        conn.execute(
            text(f"""
                INSERT INTO realconsumption (building_name, target_date, {','.join([f'h{i}' for i in range(24)])})
                VALUES (:building_name, :target_date, {','.join([f':h{i}' for i in range(24)])})
                ON CONFLICT (building_name, target_date) DO UPDATE SET
                {', '.join([f'h{i} = EXCLUDED.h{i}' for i in range(24)])}
            """),
            {
                "building_name": building_name,
                "target_date": target_date,
                **{f"h{i}": real_values[i] for i in range(24)}
            }
        )

    return jsonify({"status": f"Predictie si consum real salvate pentru {building_name} in data de {target_date}"}), 200


@swag_from({
    'tags': ['prediction'],
    'description': 'Endpoint that returns my prediction for a day from the db',
    'parameters': [
        {
            'name': 'building_name',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'Numele cladirii'
        },
        {
            'name': 'target_date',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'Data pentru care se doreste predictia (format YYYY-MM-DD)'
        }
    ],
    'responses': {
        200: {
            'description': 'Returneaza predictia pe 24h pentru cladirea si data specificata'
        },
        404: {
            'description': 'Predictia nu exista in baza de date'
        }
    }
})
@app.route('/prediction/<string:building_name>/<string:target_date>', methods=['GET'])
def get_prediction(building_name, target_date):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT * FROM predictions WHERE building_name = :name AND target_date = :date
        """), {"name": building_name, "date": target_date})
        row = result.fetchone()
        if not row:
            return jsonify({"error": "Nu exista predictie"}), 404
        data = {
            "building_name": row[0],
            "target_date": str(row[1]),
            "predicted": {f"h{i}": row[i + 2] for i in range(24)}
        }
        return jsonify(data), 200


@swag_from({
    'tags': ['real-consumption'],
    'description': 'Endpoint that returns the real cosumption for a day from the db',
    'parameters': [
        {
            'name': 'building_name',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'Numele cladirii'
        },
        {
            'name': 'target_date',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'Data pentru care se doreste consumul real (format YYYY-MM-DD)'
        }
    ],
    'responses': {
        200: {
            'description': 'Returneaza consumul real pe 24h pentru cladirea si data specificata'
        },
        404: {
            'description': 'Consumul real nu exista in baza de date'
        }
    }
})
@app.route('/real-consumption/<string:building_name>/<string:target_date>', methods=['GET'])
def get_real_consumption(building_name, target_date):
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT * FROM realconsumption WHERE building_name = :name AND target_date = :date
        """), {"name": building_name, "date": target_date})
        row = result.fetchone()
        if not row:
            return jsonify({"error": "Nu exista predictie"}), 404
        data = {
            "building_name": row[0],
            "target_date": str(row[1]),
            "real_consumption": {f"h{i}": row[i + 2] for i in range(24)}
        }
        return jsonify(data), 200


if __name__ == '__main__':
    app.run(debug=True)
