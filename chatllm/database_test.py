from sqlalchemy import Text, create_engine, Column, Integer, String
#from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.sql import select


DATABASE_URL = "mysql+pymysql://root:@localhost/mentalSathi"
engine = create_engine(DATABASE_URL, echo=True)  # echo=True for SQL query logging

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Define the ChatbotChat model
class ChatbotChat(Base):
    __tablename__ = "chatbot_chat"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), index=True)
    context = Column(Text)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_specific_message(
        
):
    try:
        stmt = select(ChatbotChat.context).where(
            (ChatbotChat.user_id == 'new@gmail.com')
        )
        session = Session(engine)
        #result = session.execute(stmt).scalar_one_or_none()  # single value or None
        results = session.execute(stmt)
        for result in results:
            print(f'The result : {result}')
        if result:
            print(f'This is the result : {result}')
            return {"message": result}
        return {"error": "Message not found"}
    except Exception as e:
        return {"error": str(e)} 


if __name__ == "__main__":
    get_specific_message()
