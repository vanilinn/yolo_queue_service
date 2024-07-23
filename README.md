# Сервис для детекции и классификации объектов на видеопотоках и изображениях из очереди RabbitMQ

## Описание

Этот сервис осуществляет детекцию и классификацию объектов на видеопотоках и изображениях из очереди RabbitMQ. Пропорция использования видеопотоков и сообщений из очереди задается параметром `SOURCE_RATIO`. Сервис поддерживает три режима работы, управляемые параметром `PROCESS_STATE`:
- `PROCESS_STATE = 1`: Сохранить изображение с аннотациями.
- `PROCESS_STATE = 2`: Вырезать все объекты.
- `PROCESS_STATE = 0`: Пропускать кадры.

В параметре `CLASSES` содержится список классов модели YOLO, которые необходимо искать на поступающих данных.

## Использованные технологии

- **Python**: Основной язык программирования.
- **Flask**: Веб-фреймворк для создания REST API.
- **OpenCV**: Библиотека для обработки изображений и видеопотоков.
- **Ultralytics YOLO**: Модель YOLOv10 для детекции объектов.
- **Pika**: Библиотека для работы с RabbitMQ.
- **NumPy**: Библиотека для работы с массивами данных.
- **Threading**: Модуль для многопоточной обработки.