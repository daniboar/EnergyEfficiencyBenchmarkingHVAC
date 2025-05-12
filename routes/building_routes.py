from flask import Blueprint
from flasgger import swag_from

from services.building_service import get_all_buildings

building_bp = Blueprint('building', __name__)


@swag_from({
    'tags': ['building'],
    'description': 'Endpoint care imi aduce toate cladirile',
    'responses': {
        200: {
            'description': 'Returneaza toate cladirile din db'
        }
    }})
@building_bp.route('/buildings', methods=['GET'])
def get_buildings():
    return get_all_buildings()
