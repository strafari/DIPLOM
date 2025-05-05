from __future__ import annotations
import requests
import os
import cv2
import torch
import argparse
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")  # URL FastAPI
SEAT_ID     = int(os.getenv("SEAT_ID", 5))                      # id места в БД
DEVICE_KEY  = os.getenv("DEVICE_KEY", "ojyntHWGrul_idmZAJWpG8osDdL56QgVpZ6IcuxgwwY=")                      
SEND_EVERY  = 1 

ENABLE_GUI = os.getenv("ENABLE_GUI", "0") == "1"

def safe_imshow(winname: str, img):
    if ENABLE_GUI:
        cv2.imshow(winname, img)

def safe_wait_key(delay: int = 1) -> int:
    return cv2.waitKey(delay) if ENABLE_GUI else -1

# ── 0. НАСТРОЙКИ ────────────────────────────────────────────────────────────
MODEL_PATH = "yolov8n-pose.pt"
CONF_TH    = 0.30     # YOLO confidence
KPT_TH     = 0.15     # Keypoint confidence threshold

# доли кадра для центрального ROI: (xmin, ymin, xmax, ymax)
DEFAULT_ROI_RATIO = (0.25, 0.20, 0.75, 0.80)

AREA_MIN_R, AREA_MAX_R = 0.02, 1.0   # доля площади бокса от кадра
AR_MIN, AR_MAX         = 0.3, 3.0    # соотношение сторон бокса

# пороги пересечения
ROI_BOX_MIN   = 0.30   # ≥30 % бокса внутри ROI
ROI_COVER_MIN = 0.30   # ≥30 % ROI покрыто боксом

DEBUG_RAW    = True
DEBUG_REASON = True
PRINT_EVERY  = 30
MODEL_PATH = "yolov8n-pose.pt"
CONF_TH    = 0.30     # YOLO confidence
KPT_TH     = 0.15     # Keypoint confidence threshold

# доли кадра для центрального ROI: (xmin, ymin, xmax, ymax)
DEFAULT_ROI_RATIO = (0.25, 0.20, 0.75, 0.80)

AREA_MIN_R, AREA_MAX_R = 0.02, 1.0   # доля площади бокса от кадра
AR_MIN, AR_MAX         = 0.3, 3.0    # соотношение сторон бокса

DEBUG_RAW    = True
DEBUG_REASON = True
PRINT_EVERY  = 30

# ── 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ────────────────────────────────────────────

def open_source(source: str | int = 0) -> cv2.VideoCapture:
    """Открываем камеру/файл. Перебираем индексы 0‑4."""
    if isinstance(source, str) and not source.isdigit():
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть источник: {source}")
        print(f"✓ Открыт источник: {source}")
        return cap

    base_idx = int(source)
    backends = [cv2.CAP_DSHOW] if os.name == "nt" else [0]
    for backend in backends:
        for idx in range(base_idx, base_idx + 5):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                print(f"✓ Камера открыта (index={idx}, backend={backend})")
                return cap
    raise RuntimeError("Не удалось открыть камеру ни по одному индексу (0‑4).")


def build_roi_poly(frame_shape: tuple[int, int], roi_ratio: tuple[float, float, float, float]) -> Polygon:
    """Создаём Polygon ROI на основе долей кадра."""
    h, w = frame_shape[:2]
    x1 = int(roi_ratio[0] * w)
    y1 = int(roi_ratio[1] * h)
    x2 = int(roi_ratio[2] * w)
    y2 = int(roi_ratio[3] * h)
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def is_human_kpts(kconf: torch.Tensor) -> bool:
    """Возвращает True, если набор ключевых точек указывает на человека.
    Требуем ≥2 лицевых и оба плеча с conf > KPT_TH."""
    # YOLOv8‑pose порядок (17 kpts): 0 nose, 1 l‑eye, 2 r‑eye, 5 l‑shoulder, 6 r‑shoulder
    facial_idx     = [0, 1, 2]
    shoulder_idx   = [5, 6]
    facial_ok   = (kconf[facial_idx]   > KPT_TH).sum() >= 2
    shoulders_ok = (kconf[shoulder_idx] > KPT_TH).sum() == 2
    return facial_ok and shoulders_ok

# ── сеть: отправить 0 / 1 на сервер ────────────────────
def send_status_to_backend(status: int) -> None:
    """
    PUT /device/seats/{SEAT_ID}/status  {"seat_status": status}
    """
    url = f"{BACKEND_URL}/device/seats/{SEAT_ID}/status"
    headers = {"X-Device-Key": DEVICE_KEY} if DEVICE_KEY else {}
    try:
        r = requests.put(url, json={"seat_status": status},
                         headers=headers, timeout=2)
        r.raise_for_status()
        print(f"✓ sent seat {SEAT_ID} → {status}")
    except Exception as e:
        # не прерываем детекцию, если сеть упала
        print(f"✗ send error: {e}")

# ── 2. ОСНОВНАЯ ФУНКЦИЯ ────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser("Seat monitor – central ROI human detector")
    ap.add_argument("--source", default=0, help="Camera index, video file or RTSP url (default: 0)")
    ap.add_argument("--roi_ratio", default="0.25,0.20,0.75,0.80", help="ROI as xmin,ymin,xmax,ymax in 0‑1 (default central box)")
    args = ap.parse_args()

    # парсим roi_ratio
    roi_ratio = tuple(map(float, args.roi_ratio.split(",")))  # type: ignore[arg‑type]
    if len(roi_ratio) != 4 or not all(0.0 <= v <= 1.0 for v in roi_ratio):
        raise ValueError("roi_ratio должно быть 4 числа 0‑1, разделённых запятой")

    torch.set_num_threads(1)
    model = YOLO(MODEL_PATH)

    cap = open_source(args.source)

    # читаем первый кадр, чтобы получить размер и сформировать ROI
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Не удалось прочитать первый кадр.")
    roi_poly = build_roi_poly(frame.shape, roi_ratio)
    roi_area = roi_poly.area

    frame_id = 0
    
    
    prev_status = None          

    # ── 3. ЦИКЛ ОБРАБОТКИ ─────────────────────────────────────────────────
    while ok:
        frame_id += 1
        h_img, w_img = frame.shape[:2]
        img_area = w_img * h_img

        # 3.1 YOLO‑инференс
        res = model.predict(
            frame,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            half=torch.cuda.is_available(),
            imgsz=640,
            conf=CONF_TH,
            iou=0.45,
            verbose=False,
        )[0]

        valid_boxes, reasons = [], []

        # 3.2 Фильтрация
        for i, box in enumerate(res.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            aspect = w / max(h, 1)

            area_r  = (w * h) / img_area
            kconf   = res.keypoints.conf[i]
            kpt_ok  = is_human_kpts(kconf)

            box_poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            inter_area = roi_poly.intersection(box_poly).area
            overlap_to_box = inter_area / max(box_poly.area, 1)
            overlap_to_roi = inter_area / max(roi_area, 1)
                        # прохождение ROI, если выполняется хотя бы одно условие
            in_roi = (overlap_to_box >= ROI_BOX_MIN) or (overlap_to_roi >= ROI_COVER_MIN)

            fail = []
            if not (AR_MIN < aspect < AR_MAX):
                fail.append("AR")
            if not (AREA_MIN_R < area_r < AREA_MAX_R):
                fail.append("AREA")
            if not kpt_ok:
                fail.append("KPTS")
            if not in_roi:
                fail.append("ROI")

            if fail:
                reasons.append(",".join(fail))
            else:
                valid_boxes.append((x1, y1, x2, y2))

        seat_status = 1 if len(valid_boxes) else 0
        
        # 3.3 Отладка
        if DEBUG_REASON and frame_id % PRINT_EVERY == 0:
            print(f"[{frame_id}] Отброшено:", reasons)

        # 3.4 Визуализация
        annotated = frame.copy()
        for (x1, y1, x2, y2) in valid_boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # ROI маска
        overlay = annotated.copy()
        # рамка ROI (тонкая белая линия)
        x1, y1, x2, y2 = map(int, roi_poly.bounds)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.fillConvexPoly(overlay, np.array(roi_poly.exterior.coords, dtype="int32"), (255, 255, 255))
        annotated = cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0)

        cv2.putText(annotated, f"People in ROI: {len(valid_boxes)}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow("Seat monitor – center ROI", annotated)

        if DEBUG_RAW:
            dbg = frame.copy()
            for b in res.boxes.xyxy.cpu().numpy():
                cv2.rectangle(dbg, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 1)
            cv2.imshow("DEBUG raw boxes", dbg)

        # следующий кадр / выход
        if cv2.waitKey(1) & 0xFF == 27:
            break
        ok, frame = cap.read()
        
        
            # --- отправляем --------------------------------------------------
    # отправляем только при изменении либо каждые SEND_EVERY кадров
        if frame_id == 1:
            prev_status = None        # объявим в первый же цикл

        if seat_status != prev_status or frame_id % SEND_EVERY == 0:
            send_status_to_backend(seat_status)
            prev_status = seat_status


    # ── 4. ЧИСТКА ─────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
