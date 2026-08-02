import io
import os
import tempfile
from collections import Counter

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from skimage import color, filters, measure, morphology
from telebot import TeleBot
from telebot.types import InputMediaPhoto
from ultralytics import YOLO


# ============================================================
# НАСТРОЙКА
# ============================================================

load_dotenv()

API_TOKEN = os.getenv("BOT_TOKEN")

if not API_TOKEN:
    raise RuntimeError(
        "Переменная окружения BOT_TOKEN не установлена. "
        "Добавьте BOT_TOKEN в файл .env."
    )

bot = TeleBot(API_TOKEN)

SEGMENTATION_MODEL_PATH = "yolov8n-seg.pt"
POSE_MODEL_PATH = "yolov8n-pose.pt"

DETECTION_CONFIDENCE = 0.5
POSE_CONFIDENCE = 0.3


# ============================================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ============================================================

try:
    # Модель сегментации загружается один раз при запуске.
    model = YOLO(SEGMENTATION_MODEL_PATH)

    # Модель определения позы человека.
    pose_model = YOLO(POSE_MODEL_PATH)

except Exception as error:
    raise SystemExit(f"Ошибка загрузки моделей YOLO: {error}")


# ============================================================
# СОСТОЯНИЯ ПОЛЬЗОВАТЕЛЕЙ
# ============================================================

WAITING_DETECT_PHOTO = "waiting_detect_photo"
WAITING_SEGMENT_PHOTO = "waiting_segment_photo"

# Состояния хранятся в памяти:
# {(chat_id, user_id): "waiting_detect_photo"}
user_states = {}


def get_user_key(message):
    """Возвращает уникальный ключ пользователя в чате."""
    return message.chat.id, message.from_user.id


def get_state(message):
    """Возвращает текущее состояние пользователя."""
    return user_states.get(get_user_key(message))


def set_state(message, state):
    """Устанавливает состояние пользователя."""
    user_states[get_user_key(message)] = state


def clear_state(message):
    """Удаляет состояние пользователя."""
    user_states.pop(get_user_key(message), None)


# ============================================================
# ЦВЕТА ОБЪЕКТОВ
# ============================================================

# Цвета записаны в формате RGB.
COLORS = {
    "person": (255, 0, 0),
    "car": (0, 80, 255),
    "dog": (0, 200, 0),
    "cat": (255, 140, 0),
    "bicycle": (128, 0, 128),
    "motorcycle": (255, 0, 255),
    "bus": (255, 215, 0),
    "truck": (0, 200, 255),
    "train": (165, 42, 42),
    "airplane": (0, 128, 128),
    "boat": (255, 20, 147),
    "bird": (255, 190, 0),
    "horse": (128, 128, 0),
    "sheep": (192, 192, 192),
    "cow": (139, 69, 19),
    "elephant": (105, 105, 105),
    "bear": (101, 67, 33),
    "zebra": (230, 230, 230),
    "giraffe": (255, 120, 0),
    "backpack": (70, 130, 180),
    "umbrella": (255, 99, 71),
    "handbag": (255, 182, 193),
    "suitcase": (160, 82, 45),
    "sports ball": (255, 160, 122),
    "bottle": (0, 200, 130),
    "cup": (123, 104, 238),
    "banana": (255, 255, 0),
    "apple": (255, 0, 0),
    "orange": (255, 165, 0),
    "broccoli": (0, 128, 0),
    "carrot": (255, 140, 0),
    "pizza": (255, 190, 0),
    "cake": (255, 192, 203),
    "chair": (139, 0, 139),
    "couch": (139, 69, 19),
    "potted plant": (34, 139, 34),
    "bed": (135, 206, 250),
    "dining table": (165, 42, 42),
    "tv": (47, 79, 79),
    "laptop": (180, 180, 180),
    "cell phone": (0, 0, 128),
    "book": (139, 69, 19),
    "clock": (255, 215, 0),
    "vase": (152, 251, 152),
    "teddy bear": (222, 184, 135),
}

DEFAULT_COLOR = (255, 255, 0)


def get_object_color(name):
    """Возвращает RGB-цвет для класса объекта."""
    return COLORS.get(name, DEFAULT_COLOR)


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def create_png_buffer(image):
    """Сохраняет NumPy-изображение в PNG BytesIO."""
    buffer = io.BytesIO()
    buffer.name = "image.png"

    Image.fromarray(image).save(buffer, format="PNG")

    buffer.seek(0)
    return buffer


def clamp_box(box, image_width, image_height):
    """Ограничивает координаты рамки размерами изображения."""
    x1, y1, x2, y2 = box

    x1 = max(0, min(int(x1), image_width))
    y1 = max(0, min(int(y1), image_height))
    x2 = max(0, min(int(x2), image_width))
    y2 = max(0, min(int(y2), image_height))

    return x1, y1, x2, y2


def send_media_safe(chat_id, media, reply_to_message_id=None):
    """
    Отправляет несколько изображений альбомом.
    Если изображение одно — отправляет его обычным сообщением.
    """
    if not media:
        return

    if len(media) == 1:
        item = media[0]

        bot.send_photo(
            chat_id=chat_id,
            photo=item.media,
            caption=item.caption,
            reply_to_message_id=reply_to_message_id,
        )
        return

    bot.send_media_group(
        chat_id=chat_id,
        media=media,
        reply_to_message_id=reply_to_message_id,
    )


def format_detection_counts(detections):
    """Формирует текст со списком и количеством объектов."""
    counts = Counter(det["name"] for det in detections)

    lines = ["Обнаруженные объекты:"]

    for name, count in counts.most_common():
        lines.append(f"• {name}: {count}")

    return "\n".join(lines)


# ============================================================
# ПРЕОБРАЗОВАНИЕ РЕЗУЛЬТАТОВ YOLO
# ============================================================

def build_detections(result, image_width, image_height):
    """
    Преобразует результат YOLO в список объектов.

    Каждый объект содержит:
    name — название класса;
    conf — уверенность;
    box — координаты рамки;
    mask — бинарную маску полного размера.
    """
    detections = []

    if result.boxes is None or result.masks is None:
        return detections

    masks_data = result.masks.data

    if masks_data is None or len(masks_data) == 0:
        return detections

    count = min(len(result.boxes), len(masks_data))

    for index in range(count):
        class_id = int(result.boxes.cls[index].item())
        confidence = float(result.boxes.conf[index].item())

        name = result.names[class_id]

        box_values = result.boxes.xyxy[index].cpu().numpy().tolist()

        box = clamp_box(
            box_values,
            image_width=image_width,
            image_height=image_height,
        )

        mask_array = masks_data[index].cpu().numpy().astype(np.float32)

        # Маска может иметь размер входа модели, поэтому приводим её
        # к размеру оригинального изображения.
        if mask_array.shape != (image_height, image_width):
            mask_array = cv2.resize(
                mask_array,
                (image_width, image_height),
                interpolation=cv2.INTER_LINEAR,
            )

        full_mask = mask_array > 0.5

        detections.append(
            {
                "name": name,
                "conf": confidence,
                "box": box,
                "mask": full_mask,
            }
        )

    return detections


# ============================================================
# СЕГМЕНТАЦИЯ
# ============================================================

def mask_to_png_bytes(mask):
    """Конвертирует бинарную маску в PNG."""
    mask_uint8 = mask.astype(np.uint8) * 255
    return create_png_buffer(mask_uint8)


def classic_segment_to_png_bytes(image, box_points):
    """Классическая сегментация: Gaussian + Sobel + Otsu."""
    try:
        image_height, image_width = image.shape[:2]

        x1, y1, x2, y2 = clamp_box(
            box_points,
            image_width,
            image_height,
        )

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        gray = color.rgb2gray(crop)
        blurred = filters.gaussian(gray, sigma=2)
        edges = filters.sobel(blurred)

        # Для изображения без перепадов яркости Otsu может не дать
        # полезного результата.
        if np.allclose(edges, edges.flat[0]):
            binary = edges > 0
        else:
            threshold = filters.threshold_otsu(edges)
            binary = edges > threshold

        binary = morphology.remove_small_objects(
            binary,
            min_size=30,
        )

        binary = morphology.closing(
            binary,
            morphology.disk(2),
        )

        binary_uint8 = binary.astype(np.uint8) * 255

        return create_png_buffer(binary_uint8)

    except Exception as error:
        print(f"Ошибка классической сегментации: {error}")
        return None


def overlay_masks(image, detections):
    """
    Накладывает полупрозрачные маски всех объектов
    на исходное изображение.
    """
    result = image.astype(np.float32).copy()
    alpha = 0.4

    for detection in detections:
        mask = detection["mask"]
        object_color = np.array(
            get_object_color(detection["name"]),
            dtype=np.float32,
        )

        result[mask] = (
            result[mask] * (1 - alpha)
            + object_color * alpha
        )

    result = np.clip(result, 0, 255).astype(np.uint8)

    pil_image = Image.fromarray(result)
    draw = ImageDraw.Draw(pil_image)

    for detection in detections:
        name = detection["name"]
        confidence = detection["conf"]
        x1, y1, _, _ = detection["box"]

        object_color = get_object_color(name)
        label = f"{name} {confidence * 100:.1f}%"

        text_y = max(0, y1 - 15)

        draw.text(
            (x1, text_y),
            label,
            fill=object_color,
        )

    return np.array(pil_image)


def draw_all_contours(image, detections):
    """
    Рисует границы всех масок YOLO на исходном изображении.
    """
    result = image.copy()

    for detection in detections:
        mask_uint8 = detection["mask"].astype(np.uint8) * 255
        object_color = get_object_color(detection["name"])

        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # result находится в RGB, поэтому цвет передаём тоже как RGB.
        cv2.drawContours(
            result,
            contours,
            contourIdx=-1,
            color=object_color,
            thickness=3,
        )

        x1, y1, _, _ = detection["box"]

        label = (
            f"{detection['name']} "
            f"{detection['conf'] * 100:.1f}%"
        )

        # Чёрная подложка для читаемости текста.
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )

        text_top = max(0, y1 - text_height - 10)
        text_bottom = max(text_height + 5, y1)

        cv2.rectangle(
            result,
            (x1, text_top),
            (x1 + text_width + 8, text_bottom),
            (0, 0, 0),
            thickness=-1,
        )

        cv2.putText(
            result,
            label,
            (x1 + 4, max(text_height, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            object_color,
            2,
            cv2.LINE_AA,
        )

    return result


def crop_object_by_mask(image, mask, box, contour_color = None):
    """
    Вырезает объект по границе маски YOLO.

    Объект остаётся цветным, фон становится белым,
    а граница объекта выделяется цветным контуром.
    """
    image_height, image_width = image.shape[:2]

    x1, y1, x2, y2 = clamp_box(
        box,
        image_width,
        image_height,
    )

    if x2 <= x1 or y2 <= y1:
        return None

    image_crop = image[y1:y2, x1:x2].copy()
    mask_crop = mask[y1:y2, x1:x2]

    if image_crop.size == 0 or mask_crop.size == 0:
        return None

    if not np.any(mask_crop):
        return None

    # Создаём белый фон.
    result = np.full_like(image_crop, 255)

    # Копируем только пиксели объекта.
    result[mask_crop] = image_crop[mask_crop]

    mask_uint8 = mask_crop.astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if contour_color != None:
        cv2.drawContours(
            result,
            contours,
            contourIdx=-1,
            color=contour_color,
            thickness=2,
        )

    return create_png_buffer(result)


# ============================================================
# АНАЛИЗ ФОРМЫ
# ============================================================

def analyze_shape(mask):
    """
    Анализирует площадь, периметр, компактность
    и эксцентриситет объекта.
    """
    labels = measure.label(mask)
    regions = measure.regionprops(labels)

    if not regions:
        return None

    # Берём самую большую связанную область.
    region = max(regions, key=lambda item: item.area)

    area = int(region.area)
    perimeter = float(region.perimeter)
    eccentricity = float(region.eccentricity)

    if perimeter > 0:
        compactness = 4 * np.pi * area / (perimeter ** 2)
    else:
        compactness = 0

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


# ============================================================
# СКЕЛЕТИЗАЦИЯ МАСОК
# ============================================================

def skeleton_to_png_bytes(mask):
    """Строит медиальную ось маски YOLO."""
    try:
        clean_mask = morphology.remove_small_objects(
            mask,
            min_size=30,
        )

        clean_mask = morphology.closing(
            clean_mask,
            morphology.disk(2),
        )

        clean_mask = morphology.remove_small_holes(
            clean_mask,
            area_threshold=50,
        )

        skeleton = morphology.skeletonize(clean_mask)

        skeleton = morphology.remove_small_objects(
            skeleton,
            min_size=10,
        )

        skeleton_uint8 = skeleton.astype(np.uint8) * 255

        return create_png_buffer(skeleton_uint8)

    except Exception as error:
        print(f"Ошибка скелетизации маски: {error}")
        return None


def classic_skeleton_to_png_bytes(image, box_points):
    """Классический скелет: Sobel + Otsu + skeletonize."""
    try:
        image_height, image_width = image.shape[:2]

        x1, y1, x2, y2 = clamp_box(
            box_points,
            image_width,
            image_height,
        )

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        gray = color.rgb2gray(crop)
        blurred = filters.gaussian(gray, sigma=2)
        edges = filters.sobel(blurred)

        if np.allclose(edges, edges.flat[0]):
            binary = edges > 0
        else:
            threshold = filters.threshold_otsu(edges)
            binary = edges > threshold

        binary = morphology.remove_small_objects(
            binary,
            min_size=30,
        )

        binary = morphology.closing(
            binary,
            morphology.disk(2),
        )

        skeleton = morphology.skeletonize(binary)
        skeleton_uint8 = skeleton.astype(np.uint8) * 255

        return create_png_buffer(skeleton_uint8)

    except Exception as error:
        print(f"Ошибка классической скелетизации: {error}")
        return None


# ============================================================
# YOLO POSE
# ============================================================

# Соединения ключевых точек COCO.
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


def pose_skeleton_to_png_bytes(image, box_points):
    """Строит анатомический скелет человека через YOLO-pose."""
    crop_path = None

    try:
        image_height, image_width = image.shape[:2]

        x1, y1, x2, y2 = clamp_box(
            box_points,
            image_width,
            image_height,
        )

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False,
        ) as temporary_file:
            crop_path = temporary_file.name

        Image.fromarray(crop).save(crop_path, format="JPEG")

        results = pose_model.predict(
            source=crop_path,
            conf=POSE_CONFIDENCE,
            verbose=False,
        )

        if not results:
            return None

        pose_result = results[0]

        if pose_result.keypoints is None:
            return None

        if pose_result.keypoints.xy is None:
            return None

        skeleton_image = np.zeros_like(crop)

        points_collection = pose_result.keypoints.xy

        confidence_collection = pose_result.keypoints.conf

        skeleton_found = False

        for person_index, keypoints in enumerate(points_collection):
            points = keypoints.cpu().numpy()

            if confidence_collection is not None:
                point_confidences = (
                    confidence_collection[person_index]
                    .cpu()
                    .numpy()
                )
            else:
                point_confidences = np.ones(len(points))

            for first_index, second_index in POSE_CONNECTIONS:
                if (
                    first_index >= len(points)
                    or second_index >= len(points)
                ):
                    continue

                first_point = points[first_index]
                second_point = points[second_index]

                first_confidence = point_confidences[first_index]
                second_confidence = point_confidences[second_index]

                if (
                    first_confidence < POSE_CONFIDENCE
                    or second_confidence < POSE_CONFIDENCE
                ):
                    continue

                if (
                    first_point[0] <= 0
                    or first_point[1] <= 0
                    or second_point[0] <= 0
                    or second_point[1] <= 0
                ):
                    continue

                cv2.line(
                    skeleton_image,
                    (
                        int(first_point[0]),
                        int(first_point[1]),
                    ),
                    (
                        int(second_point[0]),
                        int(second_point[1]),
                    ),
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                skeleton_found = True

            # Рисуем ключевые точки.
            for point, point_confidence in zip(
                points,
                point_confidences,
            ):
                if point_confidence < POSE_CONFIDENCE:
                    continue

                if point[0] <= 0 or point[1] <= 0:
                    continue

                cv2.circle(
                    skeleton_image,
                    (int(point[0]), int(point[1])),
                    3,
                    (255, 255, 255),
                    thickness=-1,
                )

        if not skeleton_found:
            return None

        return create_png_buffer(skeleton_image)

    except Exception as error:
        print(f"Ошибка YOLO-pose: {error}")
        return None

    finally:
        if crop_path and os.path.exists(crop_path):
            try:
                os.unlink(crop_path)
            except OSError as error:
                print(f"Не удалось удалить файл {crop_path}: {error}")


# ============================================================
# РЕЖИМ /SEGMENT
# ============================================================

def handle_segment_mode(message, image, detections):
    """
    Отправляет:
    1. Общее изображение с контурами.
    2. Количество объектов.
    3. Каждый объект отдельно на белом фоне.
    """
    bot.reply_to(
        message,
        format_detection_counts(detections),
    )

    contour_image = draw_all_contours(
        image,
        detections,
    )

    contour_buffer = create_png_buffer(contour_image)

    bot.send_photo(
        chat_id=message.chat.id,
        photo=contour_buffer,
        caption="Границы объектов, определённые YOLOv8-seg",
        reply_to_message_id=message.message_id,
    )

    for index, detection in enumerate(detections, start=1):
        object_color = get_object_color(detection["name"])

        object_buffer = crop_object_by_mask(
            image=image,
            mask=detection["mask"],
            box=detection["box"],
            #contour_color=object_color,
        )

        if object_buffer is None:
            continue

        caption = (
            f"Объект №{index}\n"
            f"Класс: {detection['name']}\n"
            f"Уверенность: {detection['conf'] * 100:.1f}%"
        )

        bot.send_photo(
            chat_id=message.chat.id,
            photo=object_buffer,
            caption=caption,
            reply_to_message_id=message.message_id,
        )


# ============================================================
# РЕЖИМ /DETECT
# ============================================================

def handle_detect_mode(message, image, detections):
    """Выполняет полный анализ изображения."""
    bot.reply_to(
        message,
        format_detection_counts(detections),
    )

    # Накладываем цветные маски.
    result_image = overlay_masks(
        image,
        detections,
    )

    result_buffer = create_png_buffer(result_image)

    bot.send_photo(
        chat_id=message.chat.id,
        photo=result_buffer,
        caption="Объекты и маски YOLOv8-seg",
        reply_to_message_id=message.message_id,
    )

    # Статистика формы.
    stats_lines = ["Статистика объектов:"]

    for index, detection in enumerate(detections, start=1):
        shape_info = analyze_shape(detection["mask"])

        if shape_info is None:
            stats_lines.append(
                f"• №{index} {detection['name']} "
                f"({detection['conf'] * 100:.1f}%): "
                f"форму определить не удалось"
            )
            continue

        stats_lines.append(
            f"• №{index} {detection['name']} "
            f"({detection['conf'] * 100:.1f}%): "
            f"площадь {shape_info['area']} px, "
            f"форма — {shape_info['shape']}"
        )

    stats_text = "\n".join(stats_lines)

    # Telegram ограничивает обычное сообщение 4096 символами.
    if len(stats_text) > 4000:
        stats_text = stats_text[:4000] + "\n..."

    bot.reply_to(message, stats_text)

    bot.reply_to(
        message,
        "Используемые методы:\n"
        "• YOLOv8n-seg — нейросетевая сегментация;\n"
        "• Sobel + Otsu — классическая сегментация;\n"
        "• YOLOv8n-pose — определение позы человека.",
    )

    # Сравнение сегментации.
    for index, detection in enumerate(detections, start=1):
        media = []

        classic_buffer = classic_segment_to_png_bytes(
            image,
            detection["box"],
        )

        if classic_buffer is not None:
            media.append(
                InputMediaPhoto(
                    media=classic_buffer,
                    caption=(
                        f"Объект №{index}: {detection['name']}\n"
                        "Классическая сегментация: Sobel + Otsu"
                    ),
                )
            )

        yolo_buffer = mask_to_png_bytes(
            detection["mask"],
        )

        media.append(
            InputMediaPhoto(
                media=yolo_buffer,
                caption=(
                    f"Объект №{index}: {detection['name']}\n"
                    "Современная сегментация: YOLOv8-seg"
                ),
            )
        )

        try:
            send_media_safe(
                chat_id=message.chat.id,
                media=media,
                reply_to_message_id=message.message_id,
            )
        except Exception as error:
            print(
                "Ошибка отправки сравнения сегментации "
                f"для объекта №{index}: {error}"
            )

    # Сравнение скелетов.
    for index, detection in enumerate(detections, start=1):
        media = []

        classic_skeleton_buffer = classic_skeleton_to_png_bytes(
            image,
            detection["box"],
        )

        if classic_skeleton_buffer is not None:
            media.append(
                InputMediaPhoto(
                    media=classic_skeleton_buffer,
                    caption=(
                        f"Объект №{index}: {detection['name']}\n"
                        "Классический скелет"
                    ),
                )
            )

        if detection["name"] == "person":
            modern_skeleton_buffer = pose_skeleton_to_png_bytes(
                image,
                detection["box"],
            )

            modern_caption = (
                f"Объект №{index}: person\n"
                "Анатомический скелет: YOLOv8-pose"
            )
        else:
            modern_skeleton_buffer = skeleton_to_png_bytes(
                detection["mask"],
            )

            modern_caption = (
                f"Объект №{index}: {detection['name']}\n"
                "Медиальная ось маски YOLOv8-seg"
            )

        if modern_skeleton_buffer is not None:
            media.append(
                InputMediaPhoto(
                    media=modern_skeleton_buffer,
                    caption=modern_caption,
                )
            )

        try:
            send_media_safe(
                chat_id=message.chat.id,
                media=media,
                reply_to_message_id=message.message_id,
            )
        except Exception as error:
            print(
                "Ошибка отправки сравнения скелетов "
                f"для объекта №{index}: {error}"
            )


# ============================================================
# ОБРАБОТЧИКИ КОМАНД
# ============================================================

@bot.message_handler(commands=["start", "help"])
def send_start(message):
    clear_state(message)

    bot.reply_to(
        message,
        "Бот анализа изображений.\n\n"
        "Доступные команды:\n"
        "/detect — полный анализ изображения;\n"
        "/segment — выделение и вырезание объектов;\n"
        "/cancel — отмена текущей операции.\n\n"
        "Сначала выберите команду, затем отправьте фотографию.",
    )


@bot.message_handler(commands=["detect"])
def send_detect(message):
    set_state(message, WAITING_DETECT_PHOTO)

    bot.reply_to(
        message,
        "Отправьте фотографию для полного анализа.\n"
        "Для отмены используйте /cancel.",
    )


@bot.message_handler(commands=["segment"])
def send_segment(message):
    set_state(message, WAITING_SEGMENT_PHOTO)

    bot.reply_to(
        message,
        "Отправьте фотографию.\n\n"
        "Я выделю границы объектов моделью YOLOv8-seg "
        "и отправлю каждый объект отдельно.\n\n"
        "Для отмены используйте /cancel.",
    )


@bot.message_handler(commands=["cancel"])
def send_cancel(message):
    current_state = get_state(message)

    if current_state is None:
        bot.reply_to(
            message,
            "Сейчас нет активной операции.",
        )
        return

    clear_state(message)

    bot.reply_to(
        message,
        "Операция отменена.",
    )


# ============================================================
# ОБРАБОТЧИК ФОТОГРАФИЙ
# ============================================================

@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    state = get_state(message)

    if state not in (
        WAITING_DETECT_PHOTO,
        WAITING_SEGMENT_PHOTO,
    ):
        bot.reply_to(
            message,
            "Сначала выберите режим:\n"
            "/detect — полный анализ;\n"
            "/segment — выделение объектов.",
        )
        return

    # Сбрасываем состояние сразу, чтобы одно фото обрабатывалось
    # только один раз.
    clear_state(message)

    input_path = None

    try:
        if state == WAITING_DETECT_PHOTO:
            processing_text = "Выполняю полный анализ изображения..."
        else:
            processing_text = "Выполняю сегментацию изображения..."

        bot.reply_to(message, processing_text)

        # Получаем фотографию максимального доступного размера.
        photo = message.photo[-1]

        file_info = bot.get_file(photo.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False,
        ) as temporary_file:
            temporary_file.write(downloaded_file)
            input_path = temporary_file.name

        # Загружаем оригинал в RGB.
        image = np.array(
            Image.open(input_path).convert("RGB")
        )

        image_height, image_width = image.shape[:2]

        # retina_masks=True помогает получить более точные маски
        # относительно оригинального изображения.
        results = model.predict(
            source=input_path,
            conf=DETECTION_CONFIDENCE,
            retina_masks=True,
            verbose=False,
        )

        if not results:
            bot.reply_to(
                message,
                "Модель не вернула результат.",
            )
            return

        result = results[0]

        if result.masks is None or result.boxes is None:
            bot.reply_to(
                message,
                "Объекты на изображении не обнаружены.",
            )
            return

        detections = build_detections(
            result=result,
            image_width=image_width,
            image_height=image_height,
        )

        if not detections:
            bot.reply_to(
                message,
                "Объекты на изображении не обнаружены.",
            )
            return

        if state == WAITING_SEGMENT_PHOTO:
            handle_segment_mode(
                message=message,
                image=image,
                detections=detections,
            )

        elif state == WAITING_DETECT_PHOTO:
            handle_detect_mode(
                message=message,
                image=image,
                detections=detections,
            )

        bot.reply_to(
            message,
            "Готово.\n\n"
            "Для нового анализа выберите:\n"
            "/detect или /segment",
        )

    except Exception as error:
        print(f"Ошибка обработки изображения: {error}")

        bot.reply_to(
            message,
            f"Произошла ошибка при обработке изображения:\n{error}",
        )

    finally:
        if input_path and os.path.exists(input_path):
            try:
                os.unlink(input_path)
            except OSError as error:
                print(
                    f"Не удалось удалить временный файл "
                    f"{input_path}: {error}"
                )


# ============================================================
# ОБРАБОТЧИК ТЕКСТОВЫХ СООБЩЕНИЙ
# ============================================================

@bot.message_handler(
    func=lambda message: (
        bool(message.text)
        and not message.text.startswith("/")
    )
)
def handle_text(message):
    state = get_state(message)

    if state == WAITING_DETECT_PHOTO:
        bot.reply_to(
            message,
            "Ожидаю фотографию для полного анализа.\n"
            "Для отмены используйте /cancel.",
        )

    elif state == WAITING_SEGMENT_PHOTO:
        bot.reply_to(
            message,
            "Ожидаю фотографию для сегментации.\n"
            "Для отмены используйте /cancel.",
        )

    else:
        bot.reply_to(
            message,
            "Выберите команду:\n"
            "/detect — полный анализ;\n"
            "/segment — выделение объектов.",
        )


# ============================================================
# ЗАПУСК БОТА
# ============================================================

if __name__ == "__main__":
    print("Бот запущен.")

    bot.infinity_polling(
        skip_pending=True,
        timeout=30,
        long_polling_timeout=30,
    )