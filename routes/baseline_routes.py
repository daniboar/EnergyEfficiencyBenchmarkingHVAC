from flask import Blueprint
from flasgger import swag_from

from services.baseline_service import baseline_dataset

baseline_bp = Blueprint('baseline', __name__)

@swag_from({
    'tags': ['baseline'],
    'description': 'Endpoint care imi returneaza baseline vs consumul real pe tot datasetul',
    'parameters': [
        {
            'name': 'building_name',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'Numele cladirii'
        },
    ],
    'responses': {
        200: {
            'description': 'Returneaza consumul real vs baseline pentru tot datasetul'
        },
        404: {
            'description': 'Nu exista date'
        }
    }
})
@baseline_bp.route('/baseline-dataset/<string:building_name>', methods=['GET'])
def serve_baseline_csv(building_name):
    return baseline_dataset(building_name)
