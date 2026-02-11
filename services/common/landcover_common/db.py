from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .settings import Settings


def get_engine(settings: Settings):
    return create_engine(settings.db_url, pool_pre_ping=True)


def get_session(settings: Settings):
    engine = get_engine(settings)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)
