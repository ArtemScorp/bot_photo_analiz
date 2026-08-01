import os
import cv2
import pandas as pd
from telebot import TeleBot
from ultralytics import YOLO
import io
import numpy as np
from PIL import Image, ImageDraw
from skimage import morphology, measure

# Токен из переменной окружения (не хардкодим!)
API_TOKEN = os.environ.get("BOT_TOKEN", "8714968070:AAEXAuZXUsjugRaiMHS0ICvI-YBfonouVAc")
bot = TeleBot(API_TOKEN)

# Модель YOLOv8-seg загружается ОДИН раз при старте
model = YOLO("yolov8n-seg.pt")

# Цвета по классам объектов
COLORS = {
    "person": [255, 0, 0],       # красный
    "car": [0, 0, 255],          # синий
    "dog": [0, 255, 0],          # зелёный
    "cat": [255, 165, 0],        # оранжевый
    "bicycle": [128, 0, 128],    # фиолетовый
    "bus": [255, 255, 0],        # жёлтый
    "truck": [0, 255, 255],      # голубой
    "motorcycle": [255, 0, 255], # розовый
    "train": [165, 42, 42],      # коричневый
    "airplane": [0, 128, 128],   # бирюзовый
    "boat": [255, 20, 147],      # глубокий розовый
    "bird": [255, 215, 0],       # золотой
    "horse": [128, 128, 0],      # оливковый
    "sheep": [192, 192, 192],    # серебристый
    "cow": [139, 69, 19],        # шоколадный
    "elephant": [105, 105, 105], # тёмно-серый
    "bear": [101, 67, 33],       # тёмно-коричневый
    "zebra": [255, 250, 240],    # кремовый
    "giraffe": [255, 140, 0],    # тёмно-оранжевый
    "backpack": [70, 130, 180],  # стальной синий
    "umbrella": [255, 99, 71],   # томатный
    "handbag": [255, 182, 193],  # светло-розовый
    "tie": [47, 79, 79],         # тёмно-серый
    "suitcase": [160, 82, 45],   # сиенна
    "frisbee": [255, 228, 181],  # мокасин
    "skis": [0, 100, 0],         # тёмно-зелёный
    "snowboard": [25, 25, 112],  # полуночный синий
    "sports_ball": [255, 160, 122], # светло-лососёвый
    "kite": [100, 149, 237],     # васильковый
    "baseball_bat": [222, 184, 135], # дерево
    "baseball_glove": [244, 164, 96], # персиковый
    "skateboard": [112, 128, 144], # тускло-серый
    "surfboard": [0, 191, 255],  # глубокий голубой
    "tennis_racket": [154, 205, 50], # жёлто-зелёный
    "bottle": [0, 250, 154],     # морской зелёный
    "wine_glass": [186, 85, 211], # орхидея
    "cup": [123, 104, 238],      # пурпурно-синий
    "fork": [176, 196, 222],     # светло-стальной
    "knife": [240, 230, 140],    # хаки
    "spoon": [230, 230, 250],    # лавандовый
    "bowl": [144, 238, 144],     # светло-зелёный
    "banana": [255, 255, 0],     # жёлтый
    "apple": [255, 0, 0],        # красный
    "sandwich": [210, 105, 30],  # шоколадный
    "orange": [255, 165, 0],     # оранжевый
    "broccoli": [0, 128, 0],     # зелёный
    "carrot": [255, 140, 0],     # тёмно-оранжевый
    "hot_dog": [188, 143, 143],  # розово-коричневый
    "pizza": [255, 215, 0],      # золотой
    "donut": [255, 182, 193],    # светло-розовый
    "cake": [255, 192, 203],     # розовый
    "chair": [139, 0, 139],      # тёмная орхидея
    "couch": [139, 69, 19],      # шоколадный
    "potted_plant": [34, 139, 34], # лесной зелёный
    "bed": [135, 206, 250],      # светло-голубой
    "dining_table": [165, 42, 42], # коричневый
    "toilet": [240, 255, 255],   # азурный
    "tv": [47, 79, 79],          # тёмно-серый
    "laptop": [211, 211, 211],   # светло-серый
    "mouse": [128, 128, 128],    # серый
    "remote": [169, 169, 169],   # тёмно-серый
    "keyboard": [220, 220, 220], # белый-дым
    "cell_phone": [0, 0, 128],   # тёмно-синий
    "microwave": [119, 136, 153], # сине-серый
    "oven": [255, 228, 196],     # белый-персиковый
    "toaster": [176, 196, 222],  # светло-стальной
    "sink": [199, 21, 133],      # малиновый
    "refrigerator": [72, 209, 204], # бирюзовый
    "book": [139, 69, 19],       # коричневый
    "clock": [255, 215, 0],      # золотой
    "vase": [152, 251, 152],     # бледно-зелёный
    "scissors": [192, 192, 192], # серебристый
    "teddy_bear": [222, 184, 135], # дерево
    "hair_drier": [255, 105, 180], # горячий розовый
    "toothbrush": [255, 255, 255], # белый
}
DEFAULT_COLOR = [255, 255, 0]  # жёлтый для неизвестных классов


def mask_to_png_bytes(mask):
    """Конвертирует бинарную маску в PNG BytesIO"""
    mask_uint8 = (mask * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(mask_uint8).save(buf, format="PNG")
    buf.seek(0)
    return buf


def analyze_shape(mask):
    """Анализ формы объекта: площадь, периметр, компактность, эксцентриситет"""
    labels = measure.label(mask)
    regions = measure.regionprops(labels)

    if not regions:
        return None

    region = regions[0]  # берём самую большую область
    area = region.area
    perimeter = region.perimeter
    compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    eccentricity = region.eccentricity

    # Определяем форму
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
    """Накладывает полупрозрачные цветные маски всех объектов на оригинал"""
    result = img.copy().astype(np.float64)

    # Накладываем полупрозрачные цветные маски
    for det in detections:
        mask = det["mask"]
        color = COLORS.get(det["name"], DEFAULT_COLOR)
        alpha = 0.4  # полупрозрачность

        # Наложение цвета с прозрачностью
        for c in range(3):  # R, G, B
            result[:, :, c] = np.where(mask, result[:, :, c] * (1 - alpha) + color[c] * alpha, result[:, :, c])

    result = result.astype(np.uint8)

    # Добавляем подписи классов через PIL
    pil_img = Image.fromarray(result)
    draw = ImageDraw.Draw(pil_img)
    for det in detections:
        name = det["name"]
        box = det["box"]
        color = tuple(COLORS.get(name, DEFAULT_COLOR))
        x1, y1, x2, y2 = box
        draw.text((x1, max(0, y1 - 15)), name, fill=color)

    return np.array(pil_img)


def skeleton_to_png_bytes(mask):
    """Скелетизация маски → PNG"""
    try:
        # Морфологическая очистка
        mask_clean = morphology.remove_small_objects(mask, min_size=30)

        # Скелетизация
        skeleton = morphology.skeletonize(mask_clean)

        # Нормализуем в 0-255
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(skeleton_uint8).save(buf, format="PNG")
        buf.seek(0)
        return buf

    except Exception as e:
        print(f"Something went wrong: {e}")
        return None


@bot.message_handler(commands=["start"])
def send_start(message):
    bot.reply_to(message, "жми: /detect")


@bot.message_handler(commands=["heh"])
def send_heh(message):
    parts = message.text.split()
    count_heh = int(parts[1]) if len(parts) > 1 else 5
    bot.reply_to(message, "he" * count_heh)


@bot.message_handler(commands=["detect"])
def send_detect(message):
    bot.reply_to(message, "Отправьте мне фото, и я определю объекты на нём.")


# Приём фото от пользователя
@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    try:
        bot.reply_to(message, "Анализирую изображение...")

        # Скачиваем фото от пользователя
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded = bot.download_file(file_info.file_path)

        input_path = "./content/input.jpg"

        with open(input_path, "wb") as f:
            f.write(downloaded)

        # Детекция + сегментация через YOLOv8-seg
        results = model.predict(input_path, conf=0.5)

        if not results or len(results[0].masks) == 0:
            bot.reply_to(message, "Объекты не обнаружены на изображении.")
            return

        # Загружаем оригинал
        img = np.array(Image.open(input_path).convert("RGB"))
        h, w = img.shape[:2]

        # Собираем детекции с масками
        detections = []
        for i, mask in enumerate(results[0].masks.data):
            name = results[0].names[int(results[0].boxes.cls[i])]
            conf = float(results[0].boxes.conf[i])
            box = [int(v) for v in results[0].boxes.xyxy[i]]

            # Маска в размере модели → масштабируем до размера оригинала
            mask_np = mask.cpu().numpy().astype(np.uint8)
            full_mask = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            full_mask = full_mask.astype(bool)

            detections.append({
                "name": name,
                "conf": conf,
                "box": box,
                "mask": full_mask,
            })

        # Подсчёт объектов по классам
        names = [d["name"] for d in detections]
        counts = pd.Series(names).value_counts()

        # Формируем текстовый ответ
        result_text = "Обнаруженные объекты:\n" + counts.to_string()

        bot.reply_to(message, result_text)

        # Накладываем полупрозрачные цветные маски
        result_img = overlay_masks(img, detections)

        buf = io.BytesIO()
        Image.fromarray(result_img).save(buf, format="PNG")
        buf.seek(0)
        bot.send_photo(message.chat.id, buf)

        # Статистика по каждому объекту
        stats_text = "Статистика объектов:\n"
        for det in detections:
            name = det["name"]
            conf = det["conf"]
            mask = det["mask"]
            shape_info = analyze_shape(mask)

            if shape_info:
                stats_text += (f"• {name} ({conf*100:.1f}%): "
                               f"площадь {shape_info['area']}px, "
                               f"форма: {shape_info['shape']}\n")
            else:
                stats_text += f"• {name} ({conf*100:.1f}%): не удалось определить форму\n"

        bot.reply_to(message, stats_text)

        # Отправляем маски каждого объекта
        bot.reply_to(message, "Маски объектов:")
        for det in detections:
            mask_buf = mask_to_png_bytes(det["mask"])
            bot.send_photo(message.chat.id, mask_buf)

        # Отправляем скелеты объектов
        bot.reply_to(message, "Скелеты объектов:")
        for det in detections:
            skeleton_buf = skeleton_to_png_bytes(det["mask"])
            if skeleton_buf:
                bot.send_photo(message.chat.id, skeleton_buf)

    except Exception as e:
        bot.reply_to(message, f"Произошла ошибка: {e}")


@bot.message_handler(commands=["segment"])
def send_segment(message):
    bot.reply_to(message, "Сегментация пока не реализована.")


if __name__ == "__main__":
    bot.polling()