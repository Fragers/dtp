# iVision 2.0 - команда шушпинчики

### Инструкция по запуску
- Создать окружение conda: conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
- Дополнительно понадобится opencv-python, albumentation, Pillow, numpy, efficientnet_pytorch, timm, yolov5(можно с torch hub)

###В рамках хакатона были протестированы несколько гипотез о поиске дтп, ноутбуки с тренировкой можно найти в папке src/train_notebooks:
- zero shot classification целых кадров с помощью clip дтп/не дтп 
- На основе детекции классификация разбитых машин и спецтехники
- Обучение собственной модели на классификацию целых кадров дтп/не дтп

### Также список наших датасетов - https://www.kaggle.com/katia2145/dataset-dtp-with-evacuator

### Основной скрипт для тестирования лежит по адресу: src/answer_notebook.ipynb

### Также в папке src есть еще несколько скриптов с различными решениями:
- full_frame_classification.py - обственной модели на классификацию целых кадров дтп/не дтп

## Итоговая accuracy на 50 видео составила 0.8 при учете, что ни одно видео не участвовало в тренировочной выборке