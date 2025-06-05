from datetime import date, datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Union
from starlette.responses import StreamingResponse
from fastapi_users import fastapi_users, FastAPIUsers
from pydantic import BaseModel, Field
from fastapi import HTTPException,File, Form, UploadFile
from fastapi import FastAPI, Request, status, Depends, Query, APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, insert, delete, update
from auth.auth import auth_backend
from auth.database import (
    User,
    get_async_session,
    session,
    User,
    Coworking,
    Booking,
    Seat,
    Event,
    EventRegistration,
    News,
)
from auth.manager import get_user_manager
from auth.schemas import UserRead, UserCreate
from fastapi_users.db import SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession
from auth.schemas import *
from typing import List
import io
import os
from uuid import uuid4
from fastapi.middleware.cors import CORSMiddleware
import subprocess, sys, os, signal
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
from sqlalchemy import select
from s3_client import *

load_dotenv()



app = FastAPI(title="Bashnya_mob")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


fastapi_users = FastAPIUsers[User, int](
    get_user_manager,
    [auth_backend],
)

app.include_router(
    fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"]
)

app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate), prefix="/auth", tags=["auth"]
)





############################ NEIRONKA

DEVICE_KEY = os.getenv("DEVICE_KEY")            # храните в .env / secrets
api_key_header = APIKeyHeader(name="X-Device-Key", auto_error=False)

async def verify_device(api_key: str = Depends(api_key_header)) -> None:
    if not DEVICE_KEY:
        # ключ не настроен → доступ запрещён всем
        raise HTTPException(500, "DEVICE_KEY not configured")
    if api_key != DEVICE_KEY:
        raise HTTPException(403, "Invalid device key")

device_router = APIRouter(
    prefix="/device",                
    tags=["device"],
    dependencies=[Depends(verify_device)]  
)


class SeatStatusUpdate(BaseModel):
    seat_status: int = Field(..., ge=0, le=1,
                             description="0 – свободно, 1 – занято")
    

_neironka_proc: subprocess.Popen | None = None

def _start_neironka():
#    запускает нейронку
    global _neironka_proc

    # путь к скрипту  
    script_path = r"C:\Users\4739310\Desktop\DIPLOM\diplom_project\neuronka\neironka.py"

    cmd = [
        sys.executable,      # то же окружение, что и у бэка
        script_path,
        "--source", os.getenv("NEIRONKA_SOURCE", "0"),
    ]

    env = os.environ.copy()
    # то, что мы добавляли в neironka.py
    env.setdefault("BACKEND_URL", os.getenv("BACKEND_URL", "http://127.0.0.1:8000"))
    #env.setdefault("SEAT_ID",     os.getenv("SEAT_ID",  ))
    env.setdefault("AUTH_TOKEN",  os.getenv("AUTH_TOKEN",  "ojyntHWGrul_idmZAJWpG8osDdL56QgVpZ6IcuxgwwY="))

    _neironka_proc = subprocess.Popen(cmd, env=env)
    print(f"✓ neironka запущена (pid={_neironka_proc.pid})")

def _stop_neironka():
    """Аккуратно гасим процесс YOLO при остановке FastAPI."""
    global _neironka_proc
    if _neironka_proc and _neironka_proc.poll() is None:  # ещё живой
        print("⏹  останавливаю neironka …")
        _neironka_proc.send_signal(signal.SIGTERM)
        try:
            _neironka_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _neironka_proc.kill()
        print("✓ neironka остановлена")
        
    
current_user = fastapi_users.current_user()

############################## USER 


@app.get("/htoya/", response_model=UserRead, tags=["user"])
async def hto_ya(user: User = Depends(current_user)):
    """Return data about the currently authenticated user."""
    user_data = UserRead(
        id=user.id,
        user_name=user.user_name,
        email=user.email,
        is_active=user.is_active,
        is_superuser=user.is_superuser,
    )
    return user_data


@app.get("/protected-route", tags=["user"])
def protected_route(user: User = Depends(current_user)):
    return f"Hello, {user.user_name}"


@app.get("/unprotected-route", tags=["user"])
def unprotected_route():
    return "Hello, anonym"


@app.get("/users/", tags=["user"])
async def get_all_users(
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):
    if not user.is_superuser:
        raise HTTPException(status_code=403, detail="Only superusers can access this endpoint")

    users = await session.execute(select(User))
    return users.scalars().all()


@app.delete("/delete_user/{id}", tags=["user"])
async def delete_user(
    id: int,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):
    if not user.is_superuser and user.id != id:
        raise HTTPException(status_code=403, detail="Недостаточно прав для удаления данного пользователя")

    db_user = await session.get(User, id)
    if not db_user:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
  
    await session.execute(delete(Booking).where(Booking.booking_user_id == id))
    await session.execute(delete(EventRegistration).where(EventRegistration.event_registration_user_id == id))
    await session.execute(delete(User).where(User.id == id))
    await session.commit()

    return {"message": "Пользователь и связанные данные успешно удалены"}


############################## COWORKING 


@app.get("/coworking/", response_model=List[dict], tags=["coworking"])
async def get_all_coworking(
    session: AsyncSession = Depends(get_async_session),
    # user: User = Depends(current_user),
):
    # if not user.is_superuser:
    #     raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю.")
    result = await session.execute(select(Coworking))
    coworking_spaces = result.scalars().all()
    return [
        {
            "coworking_id": cw.coworking_id,
            "coworking_location": cw.coworking_location,
          #  "coworking_description": cw.coworking_description,
        }
        for cw in coworking_spaces
    ]


@app.post("/coworking/", response_model=dict, tags=["coworking"])
async def create_coworking(
    coworking: CoworkingCreate,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):
    if not user.is_superuser:
        raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю.")
    stmt = insert(Coworking).values(**coworking.dict()).returning(Coworking)
    result = await session.execute(stmt)
    created_cw = result.scalar_one()
    await session.commit()
    return {
        "coworking_id": created_cw.coworking_id,
        "coworking_location": created_cw.coworking_location,
       # "coworking_description": created_cw.coworking_description,
    }


@app.put("/coworking/{coworking_id}", response_model=dict, tags=["coworking"])
async def update_coworking(
    coworking_id: int,
    coworking: CoworkingCreate,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):
    """Изменение информации о коворкинге (только суперпользователь)."""
    if not user.is_superuser:
        raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю.")

    stmt = (
        update(Coworking)
        .where(Coworking.coworking_id == coworking_id)
        .values(**coworking.dict())
        .returning(Coworking)
    )
    result = await session.execute(stmt)
    updated = result.scalar_one_or_none()
    if not updated:
        raise HTTPException(status_code=404, detail="Коворкинг не найден")
    await session.commit()
    return {
        "coworking_id": updated.coworking_id,
        "coworking_location": updated.coworking_location,
       # "coworking_description": updated.coworking_description,
    }


@app.delete("/coworking/{coworking_id}", tags=["coworking"])
async def delete_coworking(
    coworking_id: int,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):
    if not user.is_superuser:
        raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю.")

    # Все id мест в этом коворкинге
    seat_ids = (
        await session.execute(select(Seat.seat_id).where(Seat.seat_coworking_id == coworking_id))
    ).scalars().all()

    if seat_ids:
        await session.execute(delete(Booking).where(Booking.booking_seat_id.in_(seat_ids)))
        await session.execute(delete(Seat).where(Seat.seat_id.in_(seat_ids)))

    deleted = await session.execute(delete(Coworking).where(Coworking.coworking_id == coworking_id))
    if deleted.rowcount == 0:
        raise HTTPException(status_code=404, detail="Коворкинг не найден")

    await session.commit()
    return {"message": "Коворкинг и все связанные данные удалены"}


############################### BOOKING


@app.get("/bookings/", response_model=List[BookingRead], tags=["booking"])
async def get_all_bookings(
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):
    # if not user.is_superuser:
    #     raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю.")
    result = await session.execute(select(Booking))
    return result.scalars().all()

@app.get("/stats/bookings/counts", response_class=JSONResponse, tags=["stats"])
async def stats_bookings_counts(
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):

    if not user.is_superuser:
        raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю")

    stmt = (
        select(
            Coworking.coworking_location.label("coworking_location"),
            func.count(Booking.booking_id).label("count_bookings")
        )
        .join(Booking, Coworking.coworking_id == Booking.booking_seat_id, isouter=True)
        .group_by(Coworking.coworking_location)
    )

    result = await session.execute(stmt)
    rows = result.all()  # List[Row(event_name=..., count_bookings=... )]

    data: List[Dict[str, object]] = [
        {"coworking_location": r.coworking_location, "count_bookings": r.count_bookings}
        for r in rows
    ]
    return JSONResponse(content=data)


@app.get("/stats/bookings/daily", response_class=JSONResponse, tags=["stats"])
async def stats_bookings_daily(
    booking_start: date,
    booking_end: date,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):

    if not user.is_superuser:
        raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю")

    base_stmt = select(
        func.date(Booking.booking_start).label("day"),
        func.count(Booking.booking_id).label("count")
    )


    stmt = (
        base_stmt
        .group_by(func.date(Booking.booking_start))
        .order_by(func.date(Booking.booking_start))
    )

    result = await session.execute(stmt)
    rows = result.all()

    data: List[Dict[str, object]] = [
        {"day": r.day.isoformat(), "count": r.count}
        for r in rows
    ]
    return JSONResponse(content=data)




@app.post("/bookings/", response_model=BookingRead, tags=["booking"])
async def create_booking(
    booking: BookingCreate,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):
    booking_data = booking.dict()
    booking_data["booking_user_id"] = user.id
    booking_data["booking_start"] = booking.booking_start.replace(tzinfo=None)
    booking_data["booking_end"] = booking.booking_end.replace(tzinfo=None)
    stmt = insert(Booking).values(**booking_data).returning(Booking)
    result = await session.execute(stmt)
    created_booking = result.scalar_one()
    await session.commit()
    return created_booking


@app.put("/bookings/{booking_id}", response_model=BookingRead, tags=["booking"])
async def update_booking(
    booking_id: int,
    booking: BookingCreate,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):
    existing_booking = await session.get(Booking, booking_id)
    if not existing_booking:
        raise HTTPException(status_code=404, detail="Бронирование не найдено")

    if not user.is_superuser and existing_booking.booking_user_id != user.id:
        raise HTTPException(status_code=403, detail="Недостаточно прав для изменения данного бронирования")

    booking_data = booking.dict()
    booking_data["booking_user_id"] = existing_booking.booking_user_id  # owner can't be сменён
    booking_data["booking_start"] = booking.booking_start.replace(tzinfo=None)
    booking_data["booking_end"] = booking.booking_end.replace(tzinfo=None)

    stmt = (
        update(Booking)
        .where(Booking.booking_id == booking_id)
        .values(**booking_data)
        .returning(Booking)
    )
    result = await session.execute(stmt)
    updated_booking = result.scalar_one()
    await session.commit()
    return updated_booking


@app.delete("/bookings/{booking_id}", tags=["booking"])
async def delete_booking(
    booking_id: int,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):
    booking_obj = await session.get(Booking, booking_id)
    if not booking_obj:
        raise HTTPException(status_code=404, detail="Бронирование не найдено")

    if not user.is_superuser and booking_obj.booking_user_id != user.id:
        raise HTTPException(status_code=403, detail="Недостаточно прав для удаления данного бронирования")

    await session.execute(delete(Booking).where(Booking.booking_id == booking_id))
    await session.commit()
    return {"message": "Бронирование удалено"}


############################### SEATS


@app.get("/seats/", response_model=List[SeatRead], tags=["seats"])
async def get_all_seats(
    session: AsyncSession = Depends(get_async_session),
    # user: User = Depends(current_user),
):
    # if not user.is_superuser:
    #     raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю.")
    result = await session.execute(select(Seat))
    return result.scalars().all()


@app.post("/seats/", response_model=SeatRead, tags=["seats"])
async def create_seat(
    seat: SeatCreate,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):
    if not user.is_superuser:
        raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю.")
    stmt = insert(Seat).values(**seat.dict()).returning(Seat)
    result = await session.execute(stmt)
    created_seat = result.scalar_one()
    await session.commit()
    return created_seat


@device_router.get("/seats", response_model=List[int])
async def device_list_seat_ids(
    session: AsyncSession = Depends(get_async_session),
):

    result = await session.execute(select(Seat.seat_id))
    return result.scalars().all()

@device_router.put("/seats/{seat_id}/status", response_model=SeatRead)
async def update_seat_status_device(
    seat_id: int,
    body: SeatStatusUpdate,   
    session: AsyncSession = Depends(get_async_session),
):
    stmt = (
        update(Seat)
        .where(Seat.seat_id == seat_id)
        .values(seat_status=body.seat_status)
        .returning(Seat)
    )
    result = await session.execute(stmt)
    seat = result.scalar_one_or_none()
    if seat is None:
        raise HTTPException(404, "Seat not found")
    await session.commit()
    return seat
    
@app.put("/seats/{seat_id}/status", response_model=SeatRead, tags=["seats"])
async def update_seat_status(
    seat_id: int,
    body: SeatStatusUpdate,
    session: AsyncSession = Depends(get_async_session),
    # 👉 если хотите, уберите Depends(current_user), чтобы камера могла
    #    слать данные без JWT
):
    stmt = (
        update(Seat)
        .where(Seat.seat_id == seat_id)
        .values(seat_status=body.seat_status)
        .returning(Seat)
    )
    result = await session.execute(stmt)
    seat = result.scalar_one_or_none()
    if seat is None:
        raise HTTPException(404, "Seat not found")
    await session.commit()
    return seat





@app.delete("/seats/{seat_id}", tags=["seats"])
async def delete_seat(
    seat_id: int,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):
    if not user.is_superuser:
        raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю.")

    await session.execute(delete(Booking).where(Booking.booking_seat_id == seat_id))
    deleted = await session.execute(delete(Seat).where(Seat.seat_id == seat_id))
    if deleted.rowcount == 0:
        raise HTTPException(status_code=404, detail="Место не найдено")
    await session.commit()
    return {"message": "Место и связанные бронирования удалены"}


######################## EVENTS



@app.post("/events/", response_model=EventRead, tags=["events"])
async def create_event(
    event_name: str = Form(...),
    event_description: str = Form(...),
    event_date_time: datetime = Form(...),
    event_location: str = Form(...),
    event_max_seats: int = Form(...),
    photo: UploadFile = File(...),
    event_host: str = Form(...),
    event_price: str = Form(...),
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):
    if not user.is_superuser:
        raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю.")

    # 1) Загружаем картинку
    key = generate_s3_key(photo.filename)
    await upload_fileobj_to_s3(photo.file, key, photo.content_type)

    # 2) Сохраняем остальные поля + ключ
    event_data = {
        "event_name": event_name,
        "event_description": event_description,
        "event_date_time": event_date_time.replace(tzinfo=None),
        "event_location": event_location,
        "event_max_seats": event_max_seats,
        "event_photo": key,  # здесь в таблице хранится «ключ»
        "event_host": event_host,
        "event_price": event_price,
    }
    stmt = insert(Event).values(**event_data).returning(Event)
    result = await session.execute(stmt)
    created = result.scalar_one()
    await session.commit()

    # 3) Выдаём presigned URL
    presigned = await generate_presigned_url(created.event_photo)
    return EventRead(
        event_id=created.event_id,
        event_name=created.event_name,
        event_description=created.event_description,
        event_date_time=created.event_date_time,
        event_location=created.event_location,
        event_max_seats=created.event_max_seats,
        event_host=created.event_host,
        event_price=created.event_price,
        event_photo=presigned,
    )

from typing import List

@app.get("/events/", response_model=List[EventRead], tags=["events"])
async def get_all_events(
    session: AsyncSession = Depends(get_async_session),
    # user: User = Depends(current_user),  # если нужна авторизация
):
    result = await session.execute(select(Event))
    event_list = result.scalars().all()
    event_out: list[EventRead] = []

    for item in event_list:
        key = item.event_photo  # здесь именно «ключ» из колонки
        if not key:
            presigned: Optional[str] = None
        else:
            presigned = await generate_presigned_url(key)

        event_out.append(
            EventRead(
                event_id=item.event_id,
                event_name=item.event_name,
                event_description=item.event_description,
                event_date_time=item.event_date_time,
                event_location=item.event_location,
                event_max_seats=item.event_max_seats,
                event_host=item.event_host,
                event_price=item.event_price,
                event_photo=presigned,
            )
        )

    return event_out

        



@app.get("/events/{event_id}", response_model=EventRead, tags=["events"])
async def get_event_by_id(
    event_id: int,
    session: AsyncSession = Depends(get_async_session),
    # user: User = Depends(current_user),
):
    result = await session.execute(select(Event).where(Event.c.event_id == event_id))
    event_item = result.scalar_one_or_none()
    if not event_item:
        raise HTTPException(status_code=404, detail="Event not found")

    presigned = await generate_presigned_url(event_item.event_photo)
    return EventRead(
        event_id=event_item.event_id,
        event_name=event_item.event_name,
        event_description=event_item.event_description,
        event_date_time=event_item.event_date_time,
        event_location=event_item.event_location,
        event_max_seats=event_item.event_max_seats,
        event_host=event_item.event_host,
        event_price=event_item.event_price,
        event_photo=presigned,
    )
    
        
  
  



@app.delete("/events/{event_id}", tags=["events"])
async def delete_event(
    event_id: int,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):
    if not user.is_superuser:
        raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю.")

    #  бд запись, чтобы достать ключ
    result = await session.execute(select(Event).where(Event.c.event_id == event_id))
    event_item = result.scalar_one_or_none()
    if not event_item:
        raise HTTPException(status_code=404, detail="Событие не найдено")

    # удаляет объект из S3
    await delete_object_from_s3(event_item.event_photo)

 
    await session.execute(delete(Event).where(Event.c.event_id == event_id))
    await session.commit()

    return {"message": "Событие и связанное фото удалены"}



@app.put("/events/{event_id}", response_model=EventRead, tags=["events"])
async def update_event(
    event_id: int,
    event: EventCreate,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):
    if not user.is_superuser:
        raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю.")

    event_data = event.dict()
    event_data["event_date_time"] = event_data["event_date_time"].replace(tzinfo=None)

    stmt = (
        update(Event)
        .where(Event.event_id == event_id)
        .values(**event_data)
        .returning(Event)
    )
    result = await session.execute(stmt)
    updated_event = result.scalar_one_or_none()
    if not updated_event:
        raise HTTPException(status_code=404, detail="Событие не найдено")
    await session.commit()
    return updated_event


# @app.delete("/events/{event_id}", tags=["events"])
# async def delete_event(
#     event_id: int,
#     session: AsyncSession = Depends(get_async_session),
#     user: User = Depends(current_user),
# ):
#     if not user.is_superuser:
#         raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю.")

#     await session.execute(delete(EventRegistration).where(EventRegistration.event_registration_event_id == event_id))
#     deleted = await session.execute(delete(Event).where(Event.event_id == event_id))
#     if deleted.rowcount == 0:
#         raise HTTPException(status_code=404, detail="Событие не найдено")
#     await session.commit()
#     return {"message": "Событие и связанные регистрации удалены"}


# EVENT_REGISTRATIONS


@app.get("/event_registrations/", response_model=List[EventRegistrationRead], tags=["event_reg"])
async def get_all_event_registrations(
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):
    # if not user.is_superuser:
    #     raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю.")
    result = await session.execute(select(EventRegistration))
    return result.scalars().all()


@app.get("/stats/registrations/counts", response_class=JSONResponse, tags=["stats"])
async def stats_registrations_counts(
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):

    if not user.is_superuser:
        raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю")

    stmt = (
        select(
            Event.event_name.label("event_name"),
            func.count(EventRegistration.event_registration_id).label("count_regs")
        )
        .join(EventRegistration, Event.event_id == EventRegistration.event_registration_id, isouter=True)
        .group_by(Event.event_name)
    )

    result = await session.execute(stmt)
    rows = result.all()  # List[Row(event_name=..., count_regs=...)]

    # 2) Преобразуем в Python-список словарей
    data: List[Dict[str, object]] = [
        {"event_name": r.event_name, "count_regs": r.count_regs}
        for r in rows
    ]
    return JSONResponse(content=data)

@app.get("/stats/registrations/daily", response_class=JSONResponse, tags=["stats"])
async def stats_registrations_daily(
    event_reg_date_time: Optional[date] = None,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):

    if not user.is_superuser:
        raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю")

    # 1) Базовый запрос: SELECT DATE(reg_date) AS day, COUNT(reg_id) AS count
    base_stmt = select(
        func.date(EventRegistration.event_reg_date_time).label("day"),
        func.count(EventRegistration.event_registration_id).label("count")
    )

    # 2) Если переданы даты, добавляем WHERE
    if event_reg_date_time is not None:
        base_stmt = base_stmt.where(EventRegistration.event_reg_date_time >= event_reg_date_time)

    # 3) Добаавлем GROUP BY и ORDER BY
    stmt = (
        base_stmt
        .group_by(func.date(EventRegistration.event_reg_date_time))
        .order_by(func.date(EventRegistration.event_reg_date_time))
    )

    result = await session.execute(stmt)
    rows = result.all()  # List[Row(day=date_obj, count=int)]

    # 4) Собираем JSON-список
    data: List[Dict[str, object]] = [
        {"day": r.day.isoformat(), "count": r.count}
        for r in rows
    ]
    return JSONResponse(content=data)

@app.post("/event_registrations/", response_model=EventRegistrationRead, tags=["event_reg"])
async def create_event_registration(
    registration: EventRegistrationCreate,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):
    registration_data = {
        "event_registration_user_id": user.id,
        "event_registration_event_id": registration.event_id,
        "event_reg_date_time": datetime.utcnow(),
        "event_reg_email": registration.event_reg_email,
    }
    stmt = insert(EventRegistration).values(**registration_data).returning(EventRegistration)
    result = await session.execute(stmt)
    created_registration = result.scalar_one()
    await session.commit()
    return created_registration


@app.put("/event_registrations/{registration_id}", response_model=EventRegistrationRead, tags=["event_reg"])
async def update_event_registration(
    registration_id: int,
    registration: EventRegistrationCreate,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):
    """Update registration. Owner (user) may update their own, superuser – any."""
    existing_reg = await session.get(EventRegistration, registration_id)
    if not existing_reg:
        raise HTTPException(status_code=404, detail="Регистрация не найдена")
    if not user.is_superuser and existing_reg.event_registration_user_id != user.id:
        raise HTTPException(status_code=403, detail="Недостаточно прав для изменения данной регистрации")

    reg_data = {
        "event_reg_email": registration.event_reg_email,
        "event_registration_event_id": registration.event_id,
    }
    stmt = (
        update(EventRegistration)
        .where(EventRegistration.event_registration_id == registration_id)
        .values(**reg_data)
        .returning(EventRegistration)
    )
    result = await session.execute(stmt)
    updated_reg = result.scalar_one()
    await session.commit()
    return updated_reg


@app.delete("/event_registrations/{registration_id}", tags=["event_reg"])
async def delete_event_registration(
    registration_id: int,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):
    reg = await session.get(EventRegistration, registration_id)
    if not reg:
        raise HTTPException(status_code=404, detail="Регистрация не найдена")
    if not user.is_superuser and reg.event_registration_user_id != user.id:
        raise HTTPException(status_code=403, detail="Недостаточно прав для удаления данной регистрации")

    await session.execute(delete(EventRegistration).where(EventRegistration.event_registration_id == registration_id))
    await session.commit()
    return {"message": "Регистрация удалена"}


# NEWS

@app.get("/news/", response_model=List[NewsRead], tags=["news"])
async def get_all_news(
    session: AsyncSession = Depends(get_async_session),
    # user: User = Depends(current_user),   # по желанию
):
    result = await session.execute(select(News))
    news_list = result.scalars().all()
    news_out = []
    for item in news_list:
        presigned = await generate_presigned_url(item.news_photo)
        news_out.append(
            NewsRead(
                news_id=item.news_id,
                news_title=item.news_title,
                news_text=item.news_text,
                news_date=item.news_date,
                news_photo=presigned,
            )
        )
    return news_out


@app.get("/news/{news_id}", response_model=NewsRead, tags=["news"])
async def get_news_by_id(
    news_id: int,
    session: AsyncSession = Depends(get_async_session),
    # user: User = Depends(current_user),
):
    result = await session.execute(select(News).where(News.c.news_id == news_id))
    news_item = result.scalar_one_or_none()
    if not news_item:
        raise HTTPException(status_code=404, detail="Новость не найдена")

    # ключ для фотки
    presigned = await generate_presigned_url(news_item.news_photo)

    return NewsRead(
        news_id=news_item.news_id,
        news_title=news_item.news_title,
        news_text=news_item.news_text,
        news_date=news_item.news_date,
        news_photo=presigned,
    )

    


@app.post("/news/", response_model=NewsRead, tags=["news"])
async def create_news(
    news_title: str = Form(...),
    news_text: str = Form(...),
    news_date: datetime = Form(...),
    photo: UploadFile = File(...),
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):
    if not user.is_superuser:
        raise HTTPException(status_code=403, detail="Доступ запретён")

    # генерит ключ и грузит картику 
    key = generate_s3_key(photo.filename)
    await upload_fileobj_to_s3(photo.file, key, photo.content_type)

    news_data = {
        "news_title": news_title,
        "news_text": news_text,
        "news_date": news_date.replace(tzinfo=None),
        "news_photo": key, 
    }
    stmt = insert(News).values(**news_data).returning(News)
    result = await session.execute(stmt)
    created = result.scalar_one()
    await session.commit()

    # генерит url для фотки
    presigned = await generate_presigned_url(created.news_photo)


    return NewsRead(
        news_id=created.news_id,
        news_title=created.news_title,
        news_text=created.news_text,
        news_date=created.news_date,
        news_photo=presigned,
    )



@app.put("/news/{news_id}", response_model=NewsRead, tags=["news"])
async def update_news(
    news_id: int,
    news_item: NewsCreate,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):
    if not user.is_superuser:
        raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю.")

    news_data = news_item.dict()
    news_data["news_date"] = news_data["news_date"].replace(tzinfo=None)

    stmt = (
        update(News).where(News.news_id == news_id).values(**news_data).returning(News)
    )
    result = await session.execute(stmt)
    updated_news = result.scalar_one_or_none()
    if not updated_news:
        raise HTTPException(status_code=404, detail="Новость не найдена")
    await session.commit()
    return updated_news





@app.delete("/news/{news_id}", tags=["news"])
async def delete_news(
    news_id: int,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_user),
):
    if not user.is_superuser:
        raise HTTPException(status_code=403, detail="Доступ запрещён")

    result = await session.execute(select(News).where(News.c.news_id == news_id))
    news_item = result.scalar_one_or_none()
    if not news_item:
        raise HTTPException(status_code=404, detail="Новость не найдена")

    # Удаляем картинку из S3
    await delete_object_from_s3(news_item.news_photo)

    # Удаляем запись из БД
    await session.execute(delete(News).where(News.c.news_id == news_id))
    await session.commit()

    # Возвращаем простой ответ без response_model
    return {"message": "Новость и картинка удалены"}




app.include_router(device_router)

@app.on_event("startup")
async def _on_startup():
    _start_neironka()

@app.on_event("shutdown")
async def _on_shutdown():
    _stop_neironka()






#########################################СТАРЫЕ ЭВЕНТЫ

# @app.get("/events/", response_model=List[EventRead], tags=["events"])
# async def get_all_events(
#     session: AsyncSession = Depends(get_async_session),
#     user: User = Depends(current_user),
# ):
#     # if not user.is_superuser:
#     #     raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю.")
#     result = await session.execute(select(Event))
#     return result.scalars().all()

# @app.get("/events/", response_model=List[EventRead], tags=["events"])
# async def get_all_events(
#     session: AsyncSession = Depends(get_async_session),
    
#  ):
#     result = await session.execute(select(Event))
#     return result.scalars().all()

# @app.get("/events/{event_id}", response_model=EventRead, tags=["events"])
# async def get_event_by_id(
#     event_id: int,
#     session: AsyncSession = Depends(get_async_session),
# ):
#     result = await session.execute(
#         select(Event).where(Event.event_id == event_id)
#     )
#     event = result.scalar_one_or_none()
#     if event is None:
#         raise HTTPException(status_code=404, detail="Event not found")
#     return event


# @app.post("/events/", response_model=EventRead, tags=["events"])
# async def create_event(
#     event: EventCreate,
#     session: AsyncSession = Depends(get_async_session),
#     user: User = Depends(current_user),
# ):
#     if not user.is_superuser:
#         raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю.")
#     event_data = event.dict()
#     event_data["event_date_time"] = event_data["event_date_time"].replace(tzinfo=None)
#     stmt = insert(Event).values(**event_data).returning(Event)
#     result = await session.execute(stmt)
#     created_event = result.scalar_one()
#     await session.commit()
#     return created_event



################################## СТАРЫЕ НОВОСТИ


# @app.get("/news/", response_model=List[NewsRead], tags=["news"])
# async def get_all_news(
#     session: AsyncSession = Depends(get_async_session),
#     user: User = Depends(current_user),
# ):
#     # if not user.is_superuser:
#     #     raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю.")
#     result = await session.execute(select(News))
#     return result.scalars().all()

# @app.get("/news/", response_model=List[NewsRead], tags=["news"])
# async def get_all_news(
#     session: AsyncSession = Depends(get_async_session),
    
# ):
    
#     result = await session.execute(select(News))
#     return result.scalars().all()

# @app.get("/news/{news_id}", response_model=NewsRead, tags=["news"])
# async def get_news_by_id(
#     news_id: int,
#     session: AsyncSession = Depends(get_async_session),
# ):
#     result = await session.execute(
#         select(News).where(News.news_id == news_id)
#     )
#     news_item = result.scalar_one_or_none()
#     if news_item is None:
#         raise HTTPException(status_code=404, detail="News not found")
#     return news_item


# @app.post("/news/", response_model=NewsRead, tags=["news"])
# async def create_news(
#     news_item: NewsCreate,
#     session: AsyncSession = Depends(get_async_session),
#     user: User = Depends(current_user),
# ):
#     if not user.is_superuser:
#         raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю.")
#     news_data = news_item.dict()
#     news_data["news_date"] = news_data["news_date"].replace(tzinfo=None)
#     stmt = insert(News).values(**news_data).returning(News)
#     result = await session.execute(stmt)
#     created_news = result.scalar_one()
#     await session.commit()
#     return created_news



# @app.delete("/news/{news_id}", tags=["news"])
# async def delete_news(
#     news_id: int,
#     session: AsyncSession = Depends(get_async_session),
#     user: User = Depends(current_user),
# ):
#     if not user.is_superuser:
#         raise HTTPException(status_code=403, detail="Доступ разрешён только суперпользователю.")

#     deleted = await session.execute(delete(News).where(News.news_id == news_id))
#     if deleted.rowcount == 0:
#         raise HTTPException(status_code=404, detail="Новость не найдена")
#     await session.commit()
#     return {"message": "Новость удалена"}