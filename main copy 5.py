from dotenv import load_dotenv

load_dotenv()  # загружает переменные из .env
import os
import cv2
import pandas as pd
from telebot import TeleBot
from telebot.types import InputMediaPhoto
from ultralytics import YOLO
import io
import numpy as np
from PIL import Image, ImageDraw
from skimage import color, filters, morphology, measure
import tempfile

# Токен из переменной окружения
API_TOKEN = os.environ.get("BOT_TOKEN")
if not API_TOKEN:
    raise ValueError("Переменная окружения BOT_TOKEN не установлена!")

bot = TeleBot(API_TOKEN)

try:
    model = YOLO("yolov8n-seg.pt")
    pose_model = YOLO("yolov8n-pose.pt")
except Exception as e:
    print(f"Ошибка загрузки моделей: {e}")
    exit(1)

COLORS = {
    "person": [255, 0, 0],
    "car": [0, 0, 255],
    "dog": [0, 255, 0],
    "cat": [255, 165, 0],
    "bicycle": [128, 0, 128],
    "bus": [255, 255, 0],
    "truck": [0, 255, 255],
    "motorcycle": [255, 0, 255],
    "train": [165, 42, 42],
    "airplane": [0, 128, 128],
    "boat": [255, 20, 147],
    "bird": [255, 215, 0],
    "horse": [128, 128, 0],
    "sheep": [192, 192, 192],
    "cow": [139, 69, 19],
    "elephant": [105, 105, 105],
    "bear": [101, 67, 33],
    "zebra": [255, 250, 240],
    "giraffe": [255, 140, 0],
    "backpack": [70, 130, 180],
    "umbrella": [255, 99, 71],
    "handbag": [255, 182, 193],
    "tie": [47, 79, 79],
    "suitcase": [160, 82, 45],
    "frisbee": [255, 228, 181],
    "skis": [0, 100, 0],
    "snowboard": [25, 25, 112],
    "sports_ball": [255, 160, 122],
    "kite": [100, 149, 237],
    "baseball_bat": [222, 184, 135],
    "baseball_glove": [244, 164, 96],
    "skateboard": [112, 128, 144],
    "surfboard": [0, 191, 255],
    "tennis_racket": [154, 205, 50],
    "bottle": [0, 250, 154],
    "wine_glass": [186, 85, 211],
    "cup": [123, 104, 238],
    "fork": [176, 196, 222],
    "knife": [240, 230, 140],
    "spoon": [230, 230, 250],
    "bowl": [144, 238, 144],
    "banana": [255, 255, 0],
    "apple": [255, 0, 0],
    "sandwich": [210, 105, 30],
    "orange": [255, 165, 0],
    "broccoli": [0, 128, 0],
    "carrot": [255, 140, 0],
    "hot_dog": [188, 143, 143],
    "pizza": [255, 215, 0],
    "donut": [255, 182, 193],
    "cake": [255, 192, 203],
    "chair": [139, 0, 139],
    "couch": [139, 69, 19],
    "potted_plant": [34, 139, 34],
    "bed": [135, 206, 250],
    "dining_table": [165, 42, 42],
    "toilet": [240, 255, 255],
    "tv": [47, 79, 79],
    "laptop": [211, 211, 211],
    "mouse": [128, 128, 128],
    "remote": [169, 169, 169],
    "keyboard": [220, 220, 220],
    "cell_phone": [0, 0, 128],
    "microwave": [119, 136, 153],
    "oven": [255, 228, 196],
    "toaster": [176, 196, 222],
    "sink": [199, 21, 133],
    "refrigerator": [72, 209, 204],
    "book": [139, 69, 19],
    "clock": [255, 215, 0],
    "vase": [152, 251, 152],
    "scissors": [192, 192, 192],
    "teddy_bear": [222, 184, 135],
    "hair_drier": [255, 105, 180],
    "toothbrush": [255, 255, 255],
}
DEFAULT_COLOR = [255, 255, 0]


def send_media_safe(bot, chat_id, media, reply_to_message_id=None):
    """Безопасная отправка медиа: группой или одиночно"""
    if not media:
        return
    if len(media) >= 2:
        bot.send_media_group(chat_id, media, reply_to_message_id=reply_to_message_id)
    else:
        bot.send_photo(
            chat_id,
            media[0].media,
            caption=media[0].caption,
            reply_to_message_id=reply_to_message_id,
        )


def mask_to_png_bytes(mask):
    """Конвертирует бинарную маску в PNG BytesIO"""
    mask_uint8 = (mask * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(mask_uint8).save(buf, format="PNG")
    buf.seek(0)
    return buf


def classic_segment_to_png_bytes(img, box_points):
    """Классическая сегментация: Sobel + Otsu"""
    x1, y1, x2, y2 = box_points
    crop = img[y1:y2, x1:x2]

    gray = color.rgb2gray(crop)
    blurred = filters.gaussian(gray, sigma=2)
    edges = filters.sobel(blurred)

    thresh = filters.threshold_otsu(edges)
    binary = edges > thresh

    binary = morphology.remove_small_objects(binary, min_size=30)
    binary = morphology.closing(binary, morphology.disk(2))

    binary_uint8 = (binary * 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(binary_uint8).save(buf, format="PNG")
    buf.seek(0)
    return buf


def analyze_shape(mask):
    """Анализ формы объекта"""
    labels = measure.label(mask)
    regions = measure.regionprops(labels)

    if not regions:
        return None

    region = max(regions, key=lambda r: r.area)  # исправлено: берём наибольшую
    area = region.area
    perimeter = region.perimeter
    compactness = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
    eccentricity = region.eccentricity

    if compactness > 0.8:
        shape = "круглый/компактный"
    elif eccentricity > 0.8:
        shape = "вытянутый"
    elif eccentricity > 0.5:
        shape = "овальный"
    else:
        shape = "сложная форма"

    return {
        "area": area,
        "perimeter": perimeter,
        "compactness": compactness,
        "eccentricity": eccentricity,
        "shape": shape,
    }


def overlay_masks(img, detections):
    """Накладывает полупрозрачные цветные маски на оригинал"""
    result = img.copy().astype(np.float64)

    for det in detections:
        mask = det["mask"]
        obj_color = COLORS.get(det["name"], DEFAULT_COLOR)  # исправлено имя
        alpha = 0.4

        for c in range(3):
            result[:, :, c] = np.where(
                mask,
                result[:, :, c] * (1 - alpha) + obj_color[c] * alpha,
                result[:, :, c],
            )

    result = result.astype(np.uint8)

    pil_img = Image.fromarray(result)
    draw = ImageDraw.Draw(pil_img)
    for det in detections:
        name = det["name"]
        box = det["box"]
        obj_color = tuple(COLORS.get(name, DEFAULT_COLOR))  # исправлено имя
        x1, y1, x2, y2 = box
        draw.text((x1, max(0, y1 - 15)), name, fill=obj_color)

    return np.array(pil_img)


def skeleton_to_png_bytes(mask):
    """Скелетизация маски → PNG"""
    try:
        mask_clean = morphology.remove_small_objects(mask, min_size=30)
        mask_clean = morphology.closing(mask_clean, morphology.disk(2))
        mask_clean = morphology.remove_small_holes(mask_clean, area_threshold=50)

        skeleton = morphology.skeletonize(mask_clean)
        skeleton = morphology.remove_small_objects(skeleton, min_size=10)

        skeleton_uint8 = (skeleton * 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(skeleton_uint8).save(buf, format="PNG")
        buf.seek(0)
        return buf

    except Exception as e:
        print(f"skeleton_to_png_bytes error: {e}")
        return None


def classic_skeleton_to_png_bytes(img, box_points):
    """Классический скелет: Sobel + Otsu + skeletonize"""
    try:
        x1, y1, x2, y2 = box_points
        crop = img[y1:y2, x1:x2]

        gray = color.rgb2gray(crop)
        blurred = filters.gaussian(gray, sigma=2)
        edges = filters.sobel(blurred)

        thresh = filters.threshold_otsu(edges)
        binary = edges > thresh

        binary = morphology.remove_small_objects(binary, min_size=30)
        binary = morphology.closing(binary, morphology.disk(2))

        skeleton = morphology.skeletonize(binary)
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(skeleton_uint8).save(buf, format="PNG")
        buf.seek(0)
        return buf

    except Exception as e:
        print(f"classic_skeleton_to_png_bytes error: {e}")
        return None


POSE_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def pose_skeleton_to_png_bytes(img, box_points):
    """Анатомический скелет человека через YOLOv8-pose"""
    try:
        x1, y1, x2, y2 = box_points
        crop = img[y1:y2, x1:x2]

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            Image.fromarray(crop).save(tmp.name)
            crop_path = tmp.name

        try:
            results = pose_model.predict(crop_path, conf=0.3)

            if not results or results[0].keypoints is None:
                return None

            skeleton_img = np.zeros_like(crop)

            for kp in results[0].keypoints.xy:
                points = kp.cpu().numpy()
                for i, j in POSE_CONNECTIONS:
                    if i < len(points) and j < len(points):
                        pi, pj = points[i], points[j]
                        if pi[0] > 0 and pi[1] > 0 and pj[0] > 0 and pj[1] > 0:
                            cv2.line(
                                skeleton_img,
                                (int(pi[0]), int(pi[1])),
                                (int(pj[0]), int(pj[1])),
                                (255, 255, 255),
                                2,
                            )

            buf = io.BytesIO()
            Image.fromarray(skeleton_img).save(buf, format="PNG")
            buf.seek(0)
            return buf
        finally:
            os.unlink(crop_path)

    except Exception as e:
        print(f"pose_skeleton_to_png_bytes error: {e}")
        return None


@bot.message_handler(commands=["start"])
def send_start(message):
    bot.reply_to(message, "жми: /detect")


@bot.message_handler(
    func=lambda message: message.text and not message.text.startswith("/")
)
def handle_text(message):
    bot.reply_to(message, "Введите команду /start, чтобы начать работу с ботом.")


@bot.message_handler(commands=["detect"])
def send_detect(message):
    bot.reply_to(message, "Отправьте мне фото, и я определю объекты на нём.")


@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    try:
        bot.reply_to(message, "Анализирую изображение...")

        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded = bot.download_file(file_info.file_path)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(downloaded)
            input_path = tmp.name

        try:
            results = model.predict(input_path, conf=0.5)

            # ИСПРАВЛЕНО: проверка на None перед len()
            if not results or results[0].masks is None or len(results[0].masks) == 0:
                bot.reply_to(message, "Объекты не обнаружены на изображении.")
                return

            img = np.array(Image.open(input_path).convert("RGB"))
            h, w = img.shape[:2]

            detections = []
            for i, mask in enumerate(results[0].masks.data):
                name = results[0].names[int(results[0].boxes.cls[i])]
                conf = float(results[0].boxes.conf[i])
                box = [int(v) for v in results[0].boxes.xyxy[i]]

                mask_np = mask.cpu().numpy().astype(np.uint8)
                full_mask = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                full_mask = full_mask.astype(bool)

                detections.append(
                    {
                        "name": name,
                        "conf": conf,
                        "box": box,
                        "mask": full_mask,
                    }
                )

            names = [d["name"] for d in detections]
            counts = pd.Series(names).value_counts()
            result_text = "Обнаруженные объекты:\n" + counts.to_string()
            bot.reply_to(message, result_text)

            result_img = overlay_masks(img, detections)
            buf = io.BytesIO()
            Image.fromarray(result_img).save(buf, format="PNG")
            buf.seek(0)
            bot.send_photo(message.chat.id, buf)

            stats_text = "Статистика объектов:\n"
            for det in detections:
                name = det["name"]
                conf = det["conf"]
                shape_info = analyze_shape(det["mask"])

                if shape_info:
                    stats_text += (
                        f"• {name} ({conf * 100:.1f}%): "
                        f"площадь {shape_info['area']}px, "
                        f"форма: {shape_info['shape']}\n"
                    )
                else:
                    stats_text += (
                        f"• {name} ({conf * 100:.1f}%): "
                        f"не удалось определить форму\n"
                    )
            bot.reply_to(message, stats_text)

            models_info = (
                "🔬 Используемые модели:\n"
                "• YOLOv8n-seg — нейросетевая сегментация (точность ~90%+)\n"
                "• Sobel + Otsu — классическая сегментация по градиентам (~50-60%)"
            )
            bot.reply_to(message, models_info)

            # Сравнение сегментации
            for det in detections:
                name = det["name"]
                conf = det["conf"]
                box = det["box"]

                bot.reply_to(
                    message, f"📊 Сравнение сегментации для: {name} ({conf * 100:.1f}%)"
                )

                media = []

                classic_buf = classic_segment_to_png_bytes(img, box)
                if classic_buf:
                    classic_buf.seek(0)
                    media.append(
                        InputMediaPhoto(
                            media=classic_buf, caption="Классическая (Sobel + Otsu)"
                        )
                    )

                yolo_buf = mask_to_png_bytes(det["mask"])
                yolo_buf.seek(0)
                media.append(
                    InputMediaPhoto(media=yolo_buf, caption="Современная (YOLOv8-seg)")
                )

                try:
                    send_media_safe(bot, message.chat.id, media, message.message_id)
                except Exception as e:
                    print(f"Ошибка отправки сегментации: {e}")

            # Сравнение скелетов
            for det in detections:
                name = det["name"]
                conf = det["conf"]
                box = det["box"]

                bot.reply_to(
                    message, f"🦴 Сравнение скелетов для: {name} ({conf * 100:.1f}%)"
                )

                media = []

                classic_skel_buf = classic_skeleton_to_png_bytes(img, box)
                if classic_skel_buf:
                    classic_skel_buf.seek(0)
                    media.append(
                        InputMediaPhoto(
                            media=classic_skel_buf,
                            caption="Классический (Sobel + Otsu)",
                        )
                    )

                if name == "person":
                    pose_buf = pose_skeleton_to_png_bytes(img, box)
                    if pose_buf:
                        pose_buf.seek(0)
                        media.append(
                            InputMediaPhoto(
                                media=pose_buf, caption="Современный (YOLOv8-pose)"
                            )
                        )
                else:
                    yolo_skel_buf = skeleton_to_png_bytes(det["mask"])
                    if yolo_skel_buf:
                        yolo_skel_buf.seek(0)
                        media.append(
                            InputMediaPhoto(
                                media=yolo_skel_buf, caption="Современный (YOLOv8-seg)"
                            )
                        )

                try:
                    send_media_safe(bot, message.chat.id, media, message.message_id)
                except Exception as e:
                    print(f"Ошибка отправки скелетов: {e}")

        finally:
            os.unlink(input_path)

    except Exception as e:
        bot.reply_to(message, f"Произошла ошибка: {e}")


@bot.message_handler(commands=["segment"])
def send_segment(message):
    bot.reply_to(message, "Сегментация пока не реализована.")


if __name__ == "__main__":
    bot.polling()
