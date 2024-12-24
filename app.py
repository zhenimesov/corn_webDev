from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np

# Константы
UPLOAD_FOLDER = 'static/uploads/'  # Папка для загруженных файлов
MODEL_PATH = 'my_model.keras'  # Путь к сохраненной модели
IMG_HEIGHT, IMG_WIDTH = 150, 150  # Размеры изображений
CLASS_NAMES = []  # Классы зерен

# Инициализация приложения Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Загрузка модели
model = load_model(MODEL_PATH)

# Главная страница
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)  # Сохранение загруженного файла
            prediction = predict_plant(file_path)  # Предсказание класса растения
            return render_template('index.html', prediction=prediction, image_path=file.filename)  # Отображение результата
    return render_template('index.html')

# Функция для предсказания класса растения
def predict_plant(image_path):
    image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))  # Загрузка изображения с изменением размера
    image = img_to_array(image) / 255.0  # Преобразование в массив и нормализация
    image = np.expand_dims(image, axis=0)  # Добавление новой оси для батча
    predictions = model.predict(image)  # Получение предсказания модели
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]  # Класс с наибольшей вероятностью
    confidence = round(100 * np.max(predictions[0]), 2)  # Уверенность в предсказании
    return f"{predicted_class} ({confidence}% уверенности)"  # Формирование строки для отображения

if __name__ == '__main__':
    # Убедитесь, что папка для загрузок существует
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
