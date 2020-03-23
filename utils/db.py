from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
import os


class Connection:
    def __init__(self):
        db_string = os.environ["DATABASE_URL"]
        conn = create_engine(db_string)
        metadata = MetaData(conn)
        metadata.reflect()
        session = sessionmaker(conn)
        self.metadata = metadata
        self.session = session()

    def get_tables(self):
        Embedding = self.metadata.tables['embedding']
        Frame = self.metadata.tables['frame']
        Face = self.metadata.tables['face']
        VIP = self.metadata.tables['VIP']
        return Embedding, Frame, Face, VIP

    def get_session(self):
        return self.session

