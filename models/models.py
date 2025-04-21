from datetime import datetime

from sqlalchemy import MetaData, Table, Column, Integer, String, TIMESTAMP, ForeignKey, JSON, Boolean

metadata = MetaData()


user = Table(
    "user",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_name", String(200), nullable=False),
    Column("hashed_password", String(100), nullable=False),
    Column("email", String(100), nullable=False, unique=True),
    Column("registration_date", TIMESTAMP, default=datetime.utcnow),
    Column("is_superuser", Boolean, default=False ),
    Column("is_active", Boolean, default=True ),
)

coworking = Table(
    "coworking",
    metadata,
    Column("coworking_id", Integer, primary_key=True),
    Column("coworking_location", String(100), nullable=False),
    Column("coworking_description", String(100), nullable=True),
)

booking = Table(
    "booking",
    metadata,
    Column("booking_id", Integer, primary_key=True),
    Column("booking_user_id", Integer, ForeignKey("user.id"), nullable=False),
    Column("booking_seat_id", Integer, ForeignKey("seat.seat_id"), nullable=False),
    Column("booking_start", TIMESTAMP, nullable=False),
    Column("booking_end", TIMESTAMP, nullable=False),
    Column("booking_email", String(100), nullable=False, unique=True),
)

seat = Table(
    "seat",
    metadata,
    Column("seat_id", Integer, primary_key=True),
    Column("seat_coworking_id", Integer, ForeignKey("coworking.coworking_id"), nullable=False),
    Column("seat_index", Integer, nullable=False),
    Column("seat_status", Integer, nullable=False),
)

event = Table(
    "event",
    metadata,
    Column("event_id", Integer, primary_key=True),
    Column("event_name", String(100), nullable=False),
    Column("event_description", String(1000), nullable=True),
    Column("event_date_time", TIMESTAMP, nullable=False),
    Column("event_location", String(100), nullable=False),
    Column("event_max_seats", Integer, nullable=False),
)

event_registration = Table(
    "event_registration",
    metadata,
    Column("event_registration_id", Integer, primary_key=True),
    Column("event_registration_user_id", Integer, ForeignKey("user.id"), nullable=False),
    Column("event_registration_event_id", Integer, ForeignKey("event.event_id"), nullable=False),
    Column("event_reg_date_time", TIMESTAMP, nullable=False),
    Column("event_reg_email", String(100), nullable=False),
)

news = Table(
    "news",
    metadata,
    Column("news_id", Integer, primary_key=True),
    Column("news_image", String(255), nullable=False),  # URL или путь к изображению
    Column("news_text", String(5000), nullable=False),
    Column("news_date", TIMESTAMP, nullable=False),
)
