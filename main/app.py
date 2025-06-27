from flask import Flask
from flasgger import Swagger
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DB_URI, swagger_template
from routes.baseline_routes import baseline_bp
from routes.building_routes import building_bp
from routes.prediction_routes import prediction_bp
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)
Swagger(app, template=swagger_template)

db = SQLAlchemy(app)

# inregistrez toate blueprint-urile
app.register_blueprint(prediction_bp)
app.register_blueprint(building_bp)
app.register_blueprint(baseline_bp)

if __name__ == '__main__':
    app.run(debug=True)
