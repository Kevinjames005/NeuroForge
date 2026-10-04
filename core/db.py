from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime

engine = create_engine("sqlite:///./neuroforge.db")

SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True)
    filename = Column(String)
    n_channels = Column(Integer)
    sfreq = Column(Float)
    duration = Column(Float)
    upload_time = Column(
        DateTime,
        default=datetime.datetime.utcnow
    )
    status = Column(
        String,
        default="uploaded"
    )


Base.metadata.create_all(engine)