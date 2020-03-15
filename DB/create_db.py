from sqlalchemy import create_engine
from sqlalchemy import Column, String, DateTime, Integer, Float
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


db_string = "postgresql://postgres:142336@localhost:5432/VIPJETSON"

db = create_engine(db_string)

base = declarative_base()


class Frame(base):
    __tablename__ = 'frames'

    id = Column(String, primary_key=True)
    camera_id = Column(String)
    frame_url = Column(String)
    bboxes = Column(ARRAY(Integer))
    identities = Column(ARRAY(String))
    timestamp = Column(DateTime)


class VIP(base):
    __tablename__ = 'VIPs'

    id = Column(String, primary_key=True)
    name = Column(String)
    embed = Column(ARRAY(Float))

Session = sessionmaker(db)
session = Session()

base.metadata.create_all(db)
