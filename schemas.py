import uuid
from typing import Optional
from datetime import datetime
from fastapi_users import schemas
from pydantic import BaseModel, EmailStr, Field



class UserRead(schemas.BaseUser[int]):
    id: int
    user_name: str
    email: EmailStr
    is_active: bool = True
    is_superuser: bool 

    class Config:
        orm_mode = True

class UserCreate(schemas.BaseUserCreate):
    user_name: str
    password: str
    email: EmailStr
    is_active: Optional[bool] = True
    is_superuser: Optional[bool] = False

class CoworkingRead(BaseModel):
    coworking_id: int
    coworking_location: str
    coworking_description: Optional[str] = None

class CoworkingCreate(BaseModel):
    coworking_location: str
    coworking_description: Optional[str] = None


class BookingRead(BaseModel):
    booking_id: int
    booking_user_id: int
    booking_seat_id: int
    booking_start: datetime
    booking_end: datetime
    booking_email: str

class BookingCreate(BaseModel):
    booking_seat_id: int
    booking_start: datetime
    booking_end: datetime
    booking_email: str


class SeatRead(BaseModel):
    seat_id: int
    seat_coworking_id: int
    booking_id: Optional[int] = None
    seat_index: int
    seat_status: int

class SeatCreate(BaseModel):
    seat_coworking_id: int
    seat_index: int
    #seat_status: int

class SeatStatusUpdate(BaseModel):
    seat_status: int

class EventRead(BaseModel):
    event_id: int
    event_name: str
    event_description: Optional[str] = None
    event_date_time: datetime
    event_location: str
    event_max_seats: int

class EventCreate(BaseModel):
    event_name: str
    event_description: Optional[str] = None
    event_date_time: datetime
    event_location: str
    event_max_seats: int



class EventRegistrationRead(BaseModel):
    event_registration_id: int
    event_registration_user_id: int
    event_registration_event_id: int  
    event_reg_date_time: datetime
    event_reg_email: EmailStr

    class Config:
        orm_mode = True

class EventRegistrationCreate(BaseModel):
    event_id: int
    event_reg_email: str
    
    
    
class NewsRead(BaseModel):
    news_id: int
    news_image: str
    news_text: str
    news_date: datetime

class NewsCreate(BaseModel):
    news_image: str
    news_text: str
    news_date: datetime