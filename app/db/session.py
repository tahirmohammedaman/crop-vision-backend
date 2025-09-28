from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase


class Base(DeclarativeBase):
    pass


_engine_cache = {}


def get_engine(database_url: str):
    if database_url not in _engine_cache:
        _engine_cache[database_url] = create_engine(
            database_url, future=True, pool_pre_ping=True
        )
    return _engine_cache[database_url]


def get_sessionmaker(database_url: str):
    engine = get_engine(database_url)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
