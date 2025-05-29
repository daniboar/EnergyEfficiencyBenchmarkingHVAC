from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Date, Float, PrimaryKeyConstraint

Base = declarative_base()


class PrecipDepth1HR(Base):
    __tablename__ = 'precipdepth1hr'

    target_date = Column(Date, nullable=False)

    h0 = Column(Float)
    h1 = Column(Float)
    h2 = Column(Float)
    h3 = Column(Float)
    h4 = Column(Float)
    h5 = Column(Float)
    h6 = Column(Float)
    h7 = Column(Float)
    h8 = Column(Float)
    h9 = Column(Float)
    h10 = Column(Float)
    h11 = Column(Float)
    h12 = Column(Float)
    h13 = Column(Float)
    h14 = Column(Float)
    h15 = Column(Float)
    h16 = Column(Float)
    h17 = Column(Float)
    h18 = Column(Float)
    h19 = Column(Float)
    h20 = Column(Float)
    h21 = Column(Float)
    h22 = Column(Float)
    h23 = Column(Float)

    __table_args__ = (
        PrimaryKeyConstraint('target_date'),
    )
