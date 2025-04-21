from datetime import datetime
from tkinter import CHAR
from typing import AsyncGenerator
from fastapi import Depends
from fastapi_users.db import SQLAlchemyBaseUserTable, SQLAlchemyUserDatabase
from sqlalchemy import Column, String, Boolean, Integer, TIMESTAMP, ForeignKey
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from config import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER

DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
Base: DeclarativeMeta = declarative_base()


class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True)
    user_name = Column(String(200), nullable=False)
    hashed_password = Column(String(100), nullable=False)
    email = Column(String(100), nullable=False, unique=True)
    registration_date = Column(TIMESTAMP, default=datetime.utcnow)
    is_superuser = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True)
    booking = relationship("Booking", back_populates="user")  
    event_registration = relationship("EventRegistration", back_populates="user")  

class Coworking(Base):
    __tablename__ = "coworking"

    coworking_id = Column(Integer, primary_key=True)
    coworking_location = Column(String(100), nullable=False)
    coworking_description = Column(String(100), nullable=True)
    seat = relationship("Seat", back_populates="coworking") 

class Booking(Base):
    __tablename__ = "booking"

    booking_id = Column(Integer, primary_key=True)
    booking_user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    booking_seat_id = Column(Integer, ForeignKey("seat.seat_id"), nullable=False)
    booking_start = Column(TIMESTAMP, nullable=False)
    booking_end = Column(TIMESTAMP, nullable=False)
    booking_email = Column(String(100), nullable=False, unique=True)
    user = relationship("User", back_populates="booking")  
    seat = relationship("Seat", back_populates="booking")  

class Seat(Base):
    __tablename__ = "seat"

    seat_id = Column(Integer, primary_key=True)
    seat_coworking_id = Column(Integer, ForeignKey("coworking.coworking_id"), nullable=False)
    seat_index = Column(Integer, nullable=False)
    seat_status = Column(Integer, nullable=False)
    coworking = relationship("Coworking", back_populates="seat")  
    booking = relationship("Booking", back_populates="seat")  

class Event(Base):
    __tablename__ = "event"

    event_id = Column(Integer, primary_key=True)
    event_name = Column(String, nullable=False)
    event_description = Column(String, nullable=True)
    event_date_time = Column(TIMESTAMP, nullable=False)
    event_location = Column(String(100), nullable=False)
    event_max_seats = Column(Integer, nullable=False)
    event_registration = relationship("EventRegistration", back_populates="event")  
    
class EventRegistration(Base):
    __tablename__ = "event_registration"

    event_registration_id = Column(Integer, primary_key=True)
    event_registration_user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    event_registration_event_id = Column(Integer, ForeignKey("event.event_id"), nullable=False)
    event_reg_date_time = Column(TIMESTAMP, nullable=False)
    event_reg_email = Column(String(100), nullable=False)
    user = relationship("User", back_populates="event_registration")  
    event = relationship("Event", back_populates="event_registration") 


class News(Base):
    __tablename__ = "news"

    news_id = Column(Integer, primary_key=True)
    news_image = Column(String(255), nullable=False)  # URL или путь к изображению
    news_text = Column(String(5000), nullable=False)
    news_date = Column(TIMESTAMP, nullable=False)


engine = create_async_engine(DATABASE_URL)
async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session


async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, User)


async def session() -> AsyncSession:
   async with async_session_maker() as session:
        yield session