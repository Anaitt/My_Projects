from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base

engine = create_engine(
    'sqlite:///database.db',
    connect_args={'check_same_thread': False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Создание всех таблиц"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Нужна для открытия/закрытия базы данных в работе"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()