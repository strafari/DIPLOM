import os
import uuid
import aioboto3
from botocore.exceptions import ClientError
from botocore.config import Config

ENDPOINT_URL = os.getenv("ENDPOINT_URL")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
# AWS_REGION можно указать, если нужно явно задавать регион:
# AWS_REGION = os.getenv("AWS_REGION", "ru-central1")
BACKET_NAME = os.getenv("BACKET_NAME")
PRESIGNED_URL = int(os.getenv("PRESIGNED_URL", "3600"))

if not all([ENDPOINT_URL, AWS_ACCESS_KEY, AWS_SECRET_KEY, BACKET_NAME]):
    raise RuntimeError("Не заданы обязательные переменные для S3")

botocore_config = Config(
    signature_version="s3v4",
    retries={"max_attempts": 3, "mode": "standard"},
    # Некоторые приватные S3‐решения требуют force_path_style=True:
    s3={"addressing_style": "path"},
    # Если хотите указать регион явно:
    # region_name=AWS_REGION
)


async def get_s3_client():
    """
    Асинхронный менеджер для получения клиента aioboto3.
    Используйте через `async with get_s3_client() as client: ...`
    """
    session = aioboto3.Session()
    return session.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        config=botocore_config,
        # Если нужно явно передать регион, можно добавить:
        # region_name=AWS_REGION,
    )


def generate_s3_key(filename: str) -> str:
    """
    Генерируем уникальное имя для файла в бакете: uuid4.ext.
    """
    ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
    unique_id = uuid.uuid4().hex
    return f"{unique_id}.{ext}" if ext else unique_id


async def upload_fileobj_to_s3(
    file_obj, key: str, content_type: str
) -> None:
    """
    Загружает file_obj (UploadFile.file) в бакет под именем key.
    Так как бакет приватный, ACL не ставим (по умолчанию private).
    """
    async with await get_s3_client() as s3_client:
        try:
            await s3_client.upload_fileobj(
                Fileobj=file_obj,
                Bucket=BACKET_NAME,
                Key=key,
                ExtraArgs={
                    "ContentType": content_type,
                },
            )
        except ClientError as e:
            raise RuntimeError(f"Ошибка при загрузке в S3: {e}")


async def delete_object_from_s3(key: str) -> None:
    """
    Удаляет объект key из бакета.
    """
    async with await get_s3_client() as s3_client:
        try:
            await s3_client.delete_object(Bucket=BACKET_NAME, Key=key)
        except ClientError as e:
            raise RuntimeError(f"Ошибка при удалении из S3: {e}")


async def generate_presigned_url(key: str) -> str:
    """
    Генерирует presigned URL для приватного объекта key.
    По умолчанию URL живёт PRESIGNED_URL секунд.
    """
    async with await get_s3_client() as s3_client:
        try:
            url = await s3_client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": BACKET_NAME, "Key": key},
                ExpiresIn=PRESIGNED_URL,
            )
            return url
        except ClientError as e:
            raise RuntimeError(f"Не удалось сгенерировать presigned URL: {e}")
