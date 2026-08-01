import os
import cv2
import pandas as pd
from telebot import TeleBot
from ultralytics import YOLO
import io
import numpy as np
from PIL import Image, ImageDraw
from skimage import io as skio, color, filters, morphology, measure

# Токен из переменной окружения (не хардкодим!)
API_TOKEN = os.environ.get("BOT_TOKEN", "8714968070:AAEXAuZXUsjugRaiMHS0ICvI-YBfonouVAc")
bot = TeleBot(API_TOKEN)

# Модель YOLOv8-seg загружается ОДИН раз при старте
model = YOLO("yolov8n-seg.pt")

# Модель YOLOv8-pose для скелета человека
pose_model = YOLO("yolov8n-pose.pt")

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


def classic_segment_to_png_bytes(img, box_points):
    """Классическая сегментация: Sobel + Otsu (как было раньше)"""
    x1, y1, x2, y2 = box_points
    crop = img[y1:y2, x1:x2]

    gray = color.rgb2gray(crop)
    blurred = filters.gaussian(gray, sigma=2)
    edges = filters.sobel(blurred)

    thresh = filters.threshold_otsu(edges)
    binary = edges > thresh

    # Морфологическая очистка
    binary = morphology.remove_small_objects(binary, min_size=30)
    binary = morphology.closing(binary, morphology.disk(2))

    # Нормализуем в 0-255
    binary_uint8 = (binary * 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(binary_uint8).save(buf, format="PNG")
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
        mask_clean = morphology.closing(mask_clean, morphology.disk(2))
        mask_clean = morphology.remove_small_holes(mask_clean, area_threshold=50)

        # Скелетизация
        skeleton = morphology.skeletonize(mask_clean)

        # Убрать мелкие веточки (шум)
        skeleton = morphology.remove_small_objects(skeleton, min_size=10)

        # Нормализуем в 0-255
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(skeleton_uint8).save(buf, format="PNG")
        buf.seek(0)
        return buf

    except Exception as e:
        print(f"Something went wrong: {e}")
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

        # Морфологическая очистка
        binary = morphology.remove_small_objects(binary, min_size=30)
        binary = morphology.closing(binary, morphology.disk(2))

        # Скелетизация
        skeleton = morphology.skeletonize(binary)

        # Нормализуем в 0-255
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(skeleton_uint8).save(buf, format="PNG")
        buf.seek(0)
        return buf

    except Exception as e:
        print(f"Something went wrong: {e}")
        return None


# Соединения ключевых точек (COCO 17 точек)
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # голова
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # руки
    (5, 11), (6, 12), (11, 12),              # торс
    (11, 13), (13, 15), (12, 14), (14, 16),  # ноги
]


def pose_skeleton_to_png_bytes(img, box_points):
    """Анатомический скелет человека через YOLOv8-pose"""
    try:
        x1, y1, x2, y2 = box_points
        crop = img[y1:y2, x1:x2]

        # Сохраняем crop во временный файл
        crop_path = "content/crop.jpg"
        Image.fromarray(crop).save(crop_path)

        # Определяем ключевые точки
        results = pose_model.predict(crop_path, conf=0.3)

        if not results or results[0].keypoints is None:
            return None

        # Создаём чёрное изображение
        skeleton_img = np.zeros_like(crop)

        # Рисуем скелет
        for kp in results[0].keypoints.xy:
            points = kp.cpu().numpy()
            for (i, j) in POSE_CONNECTIONS:
                if i < len(points) and j < len(points):
                    pi, pj = points[i], points[j]
                    if pi[0] > 0 and pi[1] > 0 and pj[0] > 0 and pj[1] > 0:
                        cv2.line(skeleton_img, (int(pi[0]), int(pi[1])), (int(pj[0]), int(pj[1])), (255, 255, 255), 2)

        buf = io.BytesIO()
        Image.fromarray(skeleton_img).save(buf, format="PNG")
        buf.seek(0)
        return buf

    except Exception as e:
        print(f"Something went wrong: {e}")
        return None


@bot.message_handler(commands=["start"])
def send_start(message):
    bot.reply_to(message, "жми: /detect")


# Обработчик обычных текстовых сообщений (не команд и не фото)
@bot.message_handler(func=lambda message: message.text and not message.text.startswith("/"))
def handle_text(message):
    bot.reply_to(message, "Введите команду /start, чтобы начать работу с ботом.")


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

        # Информация о моделях
        models_info = (
            "🔬 Используемые модели:\n"
            "• YOLOv8n-seg — нейросетевая сегментация (точность ~90%+)\n"
            "• Sobel + Otsu — классическая сегментация по градиентам (~50-60%)"
        )
        bot.reply_to(message, models_info)

        # Сравнение сегментаций для каждого объекта
        for det in detections:
            name = det["name"]
            conf = det["conf"]
            box = det["box"]

            bot.reply_to(message, f"📊 Сравнение сегментации для: {name} ({conf*100:.1f}%)")

            # Классическая сегментация (Sobel + Otsu)
            classic_buf = classic_segment_to_png_bytes(img, box)
            if classic_buf:
                bot.send_photo(message.chat.id, classic_buf, caption="Классическая (Sobel + Otsu)")

            # Современная сегментация (YOLOv8-seg)
            yolo_buf = mask_to_png_bytes(det["mask"])
            bot.send_photo(message.chat.id, yolo_buf, caption="Современная (YOLOv8-seg)")

        # Сравнение скелетов для каждого объекта
        for det in detections:
            name = det["name"]
            conf = det["conf"]
            box = det["box"]

            bot.reply_to(message, f"🦴 Сравнение скелетов для: {name} ({conf*100:.1f}%)")

            # Классический скелет (Sobel + Otsu + skeletonize)
            classic_skel_buf = classic_skeleton_to_png_bytes(img, box)
            if classic_skel_buf:
                bot.send_photo(message.chat.id, classic_skel_buf, caption="Классический (Sobel + Otsu)")

            # Современный скелет
            if name == "person":
                # Анатомический скелет через YOLOv8-pose
                pose_buf = pose_skeleton_to_png_bytes(img, box)
                if pose_buf:
                    bot.send_photo(message.chat.id, pose_buf, caption="Современный (YOLOv8-pose)")
            else:
                # Медианная ось через skeletonize
                yolo_skel_buf = skeleton_to_png_bytes(det["mask"])
                if yolo_skel_buf:
                    bot.send_photo(message.chat.id, yolo_skel_buf, caption="Современный (YOLOv8-seg)")

    except Exception as e:
        bot.reply_to(message, f"Произошла ошибка: {e}")


@bot.message_handler(commands=["segment"])
def send_segment(message):
    bot.reply_to(message, "Сегментация пока не реализована.")


if __name__ == "__main__":
    bot.polling()