from flask import Blueprint
from flasgger import swag_from
from services.prediction_service import predict_for_day, get_prediction_for_day, get_real_consumption_for_day, \
    get_baseline_for_day, get_combined_profile

prediction_bp = Blueprint('prediction', __name__)


@swag_from({'tags': ['prediction'],
            'description': 'Endpoint care calculeaza predictia si populeaza tabelele de predictie si consum real',
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
            }})
@prediction_bp.route('/predict/<string:building_name>/<string:target_date>', methods=['POST'])
def predict(building_name, target_date):
    return predict_for_day(building_name, target_date)


@swag_from({
    'tags': ['prediction'],
    'description': 'Endpoint care imi aduce predictia pentru o zi',
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
@prediction_bp.route('/prediction/<string:building_name>/<string:target_date>', methods=['GET'])
def get_prediction(building_name, target_date):
    return get_prediction_for_day(building_name, target_date)


@swag_from({
    'tags': ['real-consumption'],
    'description': 'Endpoint care imi aduce consumul real pentru o zi',
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
@prediction_bp.route('/real-consumption/<string:building_name>/<string:target_date>', methods=['GET'])
def get_real(building_name, target_date):
    return get_real_consumption_for_day(building_name, target_date)


@swag_from({
    'tags': ['baseline'],
    'description': 'Endpoint care imi aduce baseline-ul pentru o zi',
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
            'description': 'Data pentru care se doreste baseline-ul (format YYYY-MM-DD)'
        }
    ],
    'responses': {
        200: {
            'description': 'Returneaza baseline-ul pentru cladirea si data specificata'
        },
        404: {
            'description': 'Baseline-ul nu exista in baza de date'
        }
    }
})
@prediction_bp.route('/baseline/<string:building_name>/<string:target_date>', methods=['GET'])
def get_baseline(building_name, target_date):
    return get_baseline_for_day(building_name, target_date)


@swag_from({
    'tags': ['prediction'],
    'description': 'Endpoint care imi face profilul unei cladiri',
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
            'description': 'Data pentru care se doreste baseline-ul (format YYYY-MM-DD)'
        }
    ],
    'responses': {
        200: {
            'description': 'Returneaza profilul pentru cladirea si data specificata'
        },
        404: {
            'description': 'Baseline-ul nu exista in baza de date'
        }
    }
})
@prediction_bp.route('/profile-consumption/<string:building_name>/<string:target_date>', methods=['GET'])
def get_profile(building_name, target_date):
    return get_combined_profile(building_name, target_date)
