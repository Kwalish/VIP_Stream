from sqlalchemy import Column, String, Float
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declarative_base

base = declarative_base()

class VIP(base):
    __tablename__ = 'VIPs'

    id = Column(String, primary_key=True)
    name = Column(String)
    image_url = Column(String)
    embed = Column(ARRAY(Float))
