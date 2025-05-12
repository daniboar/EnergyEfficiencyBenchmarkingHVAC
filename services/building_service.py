from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from config import DB_URI
from models.prediction_model import Base

engine = create_engine(DB_URI)
Session = sessionmaker(bind=engine)
Base.metadata.create_all(engine)


def get_all_buildings():
    with Session() as session:
        result = session.execute(text("SELECT name FROM buildings")).fetchall()
        buildings = [row[0] for row in result]
        return {"buildings": buildings}, 200
