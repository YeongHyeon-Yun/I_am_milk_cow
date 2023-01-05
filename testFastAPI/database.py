from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import text, Column, String, Text, Numeric

SQLALCHEY_DATABASE_URL = 'mysql+pymysql://ai_academy:ai_academy@intflow.serveftp.com:3306/ai_academy?charset=utf8mb4'

engine = create_engine(
  SQLALCHEY_DATABASE_URL,
  echo=True,
  pool_recycle=900,
  pool_pre_ping=True
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def get_db():
  db = SessionLocal()
  try:
    yield db
  finally:
    db.close()

Base = declarative_base()


class t_cow(Base):
  __tablename__ = 't_cow'

  cow_id = Column(Numeric(45), nullable=True, primary_key=True)
  cow_name = Column(String(30))
  cow_birth = Column(Numeric)
  cow_variety = Column(String(30))
  cow_gender = Column(String(10))
  cow_vaccination = Column(String(10))
  cow_pregnancy = Column(String(10))
  cow_milk = Column(String(10))
  cow_castration = Column(String(10))
  wish_list = Column(Numeric(10))
  user_num = Column(Numeric(5))




