import os
import pandas as pd
import shutil
from telebot import TeleBot#, apihelper
from imageai.Detection import ObjectDetection
import io
import numpy as np
from PIL import Image
from skimage import io as skio, color, filters

# Токен из переменной окружения (не хардкодим!)
API_TOKEN = os.environ.get("BOT_TOKEN", "8714968070:AAEXAuZXUsjugRaiMHS0ICvI-YBfonouVAc")
bot = TeleBot(API_TOKEN)
#apihelper.proxy = {    "https": "socks5://user:ddda263f0dce5a081fdc14d6b29113f436@127.0.0.1:1443"}


extracted_dir = os.path.join("content", "output-extracted")
if os.path.exists(extracted_dir):
    shutil.rmtree(extracted_dir)
# Модель загружается ОДИН раз при старте, а не при каждом запросе
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("model/yolov3.pt")
detector.loadModel()

def segment_to_png_bytes(imgpath):
    try:
        img = skio.imread(imgpath)
        gray = color.rgb2gray(img)
        blurred = filters.gaussian(gray, sigma=2)
        edges = filters.sobel(blurred)

        # Нормализуем в 0-255 и конвертируем в uint8
        edges_uint8 = (edges * 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(edges_uint8).save(buf, format="PNG")
        buf.seek(0)
        return buf

    except Exception as e:
        print(f"Something went wrong: {e}")
        return None


@bot.message_handler(commands=["start"])
def send_start(message):
    bot.reply_to(message, "Тут список всех команд: /heh, /detect, /segment")


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
        #print(file_info)
        downloaded = bot.download_file(file_info.file_path)
        #print(downloaded)

        input_path = "./content/input.jpg"
        output_path = "./content/output.jpg"

        with open(input_path, "wb") as f:
            f.write(downloaded)

        # Детекция объектов
        detections = detector.detectObjectsFromImage(
            input_image=input_path,
            output_image_path=output_path,
            minimum_percentage_probability=50,
            extract_detected_objects=True,
        )

        # Если объектов не найдено
        if not detections:
            bot.reply_to(message, "Объекты не обнаружены на изображении.")
            return
        #print(detections)
        # Подсчёт объектов по классам
        df = pd.DataFrame(detections[0])
        #, columns=["name", "percentage_probability", "box_points"]
        #print(df)
        resDf = df[["name", "percentage_probability", "box_points"]]
        counts = resDf["name"].value_counts()

        # Формируем текстовый ответ
        result_text = "Обнаруженные объекты:\n" + counts.to_string()

        bot.reply_to(message, result_text)

        # Отправляем фото с рамками обратно пользователю
        
        if os.path.exists(extracted_dir):
            for filename in os.listdir(extracted_dir):
                file_path = os.path.join(extracted_dir, filename)
                with open(file_path, "rb") as f:
                    bot.send_photo(message.chat.id, f)
                img_buf = segment_to_png_bytes(file_path)
                if img_buf:
                    bot.send_photo(message.chat.id, img_buf)

    except Exception as e:
        bot.reply_to(message, f"Произошла ошибка: {e}")

    finally:
        if os.path.exists(extracted_dir):
            shutil.rmtree(extracted_dir)


@bot.message_handler(commands=["segment"])
def send_segment(message):
    bot.reply_to(message, "Сегментация пока не реализована.")


if __name__ == "__main__":
    bot.polling()