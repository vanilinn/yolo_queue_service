import os
import json
import hashlib
from datetime import datetime
import urllib.request

from ultralytics import YOLO

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import torch
import cv2
import numpy as np
import pika
import threading
import queue
import time
from pathlib import Path
from flask import Flask, request, jsonify

# Параметры настройки по умолчанию
default_settings = {
    "PROCESS_STATE": 1,  # 1: сохранить изображение с аннотациями, 2: вырезать объекты, 0: пропустить изображение
    "STREAM_URLS": [],  # Добавьте ваши URL-адреса потоков здесь
    "RABBITMQ_QUEUE": 'queue',
    "SOURCE_RATIO": 0.5,
    # Процентное соотношение изображений из видеопотоков (0.5 = 50%) и из RabbitMQ (1.0 - SOURCE_RATIO = 50%)
    "CLASSES": ['person']
}

settings = default_settings.copy()

# Загрузка модели YOLOv10
model = YOLO("yolov10x.pt")

# Загрузка имен классов
model_classes = model.names

# Создание приложения Flask
app = Flask(__name__)

# Глобальные переменные для управления паузой
pause_event = threading.Event()
pause_event.set()  # Начальное состояние - не на паузе


@app.route('/update_settings', methods=['POST'])
def update_settings():
    global settings
    new_settings = request.json
    for key in new_settings:
        if key in settings:
            settings[key] = new_settings[key]
    return jsonify(settings)


@app.route('/start_processing', methods=['POST'])
def start_processing():
    thread = threading.Thread(target=run_processing)
    thread.start()
    return jsonify({"status": "processing started"})


@app.route('/pause_processing', methods=['POST'])
def pause_processing():
    pause_event.clear()
    return jsonify({"status": "processing paused"})


@app.route('/resume_processing', methods=['POST'])
def resume_processing():
    pause_event.set()
    return jsonify({"status": "processing resumed"})


def run_processing():
    # Очередь для кадров
    frame_queue = queue.Queue()

    # Словарь для хранения последних кадров с видеопотоков
    latest_frame = {url: None for url in settings["STREAM_URLS"]}

    # Потоковое событие для остановки захвата кадров
    stop_event = threading.Event()

    # Запуск потоков захвата кадров
    capture_threads = []
    for stream_url in settings["STREAM_URLS"]:
        t = threading.Thread(target=capture_frames,
                             args=(stream_url, frame_queue, stop_event, latest_frame, pause_event))
        t.start()
        capture_threads.append(t)

    # Запуск потока для получения изображений из RabbitMQ
    rabbitmq_thread = threading.Thread(target=receive_images_from_rabbitmq,
                                       args=(settings["RABBITMQ_QUEUE"], frame_queue, pause_event))
    rabbitmq_thread.start()

    # Запуск потока обработки кадров
    worker_thread = threading.Thread(target=frame_worker, args=(frame_queue, latest_frame, pause_event))
    worker_thread.start()

    # Ожидание завершения работы
    try:
        worker_thread.join()
    finally:
        stop_event.set()
        for t in capture_threads:
            t.join()
        frame_queue.put((None, None))


# Функция для обработки кадров с использованием YOLOv10 и фильтрации по классам
def process_frame(frame, source, save_path):
    # print(f"Processing frame from {source} with YOLO")
    results = model(frame)
    filtered_results = []
    for result in results:
        for i, box in enumerate(result.boxes.xyxy):
            if model_classes[int(result.boxes.cls[i])] in settings["CLASSES"]:
                filtered_results.append((box, result.boxes.conf[i], result.boxes.cls[i]))

    if settings["PROCESS_STATE"] == 0:
        return None

    annotated_frame = frame.copy()

    if settings["PROCESS_STATE"] == 1:
        for box, conf, cls in filtered_results:
            label = f'{model_classes[int(cls)]} {conf:.2f}'
            cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
        save_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path / 'annotated' / f"{source}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"),
                    annotated_frame)
        # print(f"Saved annotated frame from {source} to {save_path}")

    elif settings["PROCESS_STATE"] == 2:
        save_path.mkdir(parents=True, exist_ok=True)
        for i, (box, conf, cls) in enumerate(filtered_results):
            cropped_img = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            cv2.imwrite(
                str(save_path / 'cropped' / f"{source}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{i}_{model_classes[int(cls)]}.jpg"),
                cropped_img)
        # print(f"Saved cropped frame from {source} to {save_path}")
    return annotated_frame


# Функция для захвата последнего кадра из HLS потока
def capture_latest_frame(stream_url):
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"Error opening video stream or file: {stream_url}")
        return None
    ret, frame = cap.read()
    cap.release()
    return frame


# Функция для захвата кадров из HLS потока с адаптивной частотой
def capture_frames(stream_url, frame_queue, stop_event, latest_frame, pause_event):
    while not stop_event.is_set():
        pause_event.wait()  # Ожидание, пока pause_event не будет установлен
        start_time = time.time()
        frame = capture_latest_frame(stream_url)
        if frame is None:
            continue  # Пропустить, если кадр не получен

        latest_frame[stream_url] = frame
        elapsed_time = time.time() - start_time
        frame_queue.put((stream_url, frame))
        # print('################################################')
        # print(f"Captured frame from stream {stream_url}: {frame.shape}")


# Функция для получения изображений из RabbitMQ
def receive_images_from_rabbitmq(rabbitmq_queue, frame_queue, pause_event):
    def callback(ch, method, properties, body):
        pause_event.wait()  # Ожидание, пока pause_event не будет установлен
        # print("Received message from RabbitMQ")
        data = json.loads(body)
        # print(f"Message data: {data}")
        try:
            try:
                try:
                    date = datetime.strptime(data['datetime'], '%d %b %Y %H:%M:%S')
                except:
                    date = datetime.strptime(data['datetime'], '%d-%m-%Y %H:%M:%S')
            except Exception as e:
                date = datetime.strptime(data['datetime'], '%d.%m.%Y %H:%M:%S')
            dateurl = date.strftime("%Y-%m/%d")
            link = "{}/{}/{}".format(data['cam'], dateurl, data['screen'])
            linktomd5 = str(link + "some_hash")
            md5 = hashlib.md5(linktomd5.encode('utf-8')).hexdigest()
            imgpath = "https://host/{}/{}/{}".format(data['type'], md5, link)

            req = urllib.request.urlopen(imgpath)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)  # 'Load it as it is'
            if img is not None:
                frame_queue.put(("RabbitMQ", img))  # Изображения из RabbitMQ
                # print(f"Decoded and queued image from RabbitMQ: {img.shape}")
            else:
                print("Failed to decode image from RabbitMQ")
        except json.JSONDecodeError:
            print("Failed to decode JSON from message")
        except Exception as e:
            print(data)
            print(f"Error processing message: {e}")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    credentials = pika.PlainCredentials('username', 'password')
    connection = pika.BlockingConnection(pika.ConnectionParameters('host', 5672, '/', credentials))
    channel = connection.channel()
    channel.queue_declare(queue=rabbitmq_queue)  # Ensure the queue is declared with the same durability as existing
    channel.basic_consume(queue=rabbitmq_queue, on_message_callback=callback, auto_ack=False)
    print("Started consuming RabbitMQ messages")
    channel.start_consuming()


# Функция для многопоточной обработки кадров
def frame_worker(frame_queue, latest_frame, pause_event):
    stream_index = 0
    stream_frame_count = 0
    rabbitmq_frame_count = 0
    total_frame_count = 0

    while True:
        source, frame = frame_queue.get()
        if frame is None:
            break

        pause_event.wait()  # Приостановка обработки, если pause_event сброшен

        if total_frame_count == 0:
            stream_frame_count = 0
            rabbitmq_frame_count = 0

        if total_frame_count * settings["SOURCE_RATIO"] > stream_frame_count:
            if source in settings["STREAM_URLS"]:
                save_path = Path("tmp")
                processed_frame = process_frame(frame, 'stream', save_path)
                # if processed_frame is not None:
                # print(f"Processed stream frame from stream")
                stream_frame_count += 1
                total_frame_count += 1
            else:
                frame_queue.put((source, frame))  # Put back the RabbitMQ frame for future processing
        else:
            if source == 'RabbitMQ':
                save_path = Path("tmp")
                processed_frame = process_frame(frame, source, save_path)
                # if processed_frame is not None:
                # print("Processed frame from queue")
                rabbitmq_frame_count += 1
                total_frame_count += 1
            else:
                frame_queue.put((source, frame))  # Put back the stream frame for future processing

    print("Frame worker thread finished")


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=4597)
