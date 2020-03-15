from sqlalchemy import Column, String, DateTime, Integer
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declarative_base

base = declarative_base()

class Frame(base):
    __tablename__ = 'frames'

    id = Column(String, primary_key=True)
    camera_id = Column(String)
    frame_url = Column(String)
    bboxes = Column(ARRAY(Integer))
    identities = Column(ARRAY(String))
    timestamp = Column(DateTime)

