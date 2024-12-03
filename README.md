# Car Price Prediction API
A FastAPI application for predicting car prices based on Linear Regression Model. This project is developed for the ML and High Load Systems course at the Higher School of Economics.


Features
---
* **Predict Car Prices** - Submit features of a car and get a predicted selling price.

* **Batch Predictions** - Upload a CSV file or submit list of features with multiple car records and get predicted prices for each.

Requirements
---

* Python 3.9+
* FastAPI
* Uvicorn (for running the FastAPI application)
* Scikit-learn (for machine learning models)
* Pandas (for data manipulation)
* NumPy (for numerical operations)
  
Installation
----
1. Clone the repository:

```bash
$ git clone https://github.com/your-username/carprice-api.git
```
2. Install the required dependencies:

```bash
$ pip install -r requirements.txt
```

Usage
---

Run the FastAPI application using Uvicorn:

```bash
$ uvicorn main:app --reload
```

API Endpoints
---

* **POST /predict_item** - Predict the price for a single car.
* **POST /predict_items** - Predict prices for a list of cars.
* **POST /predict_file** - Predict prices from a CSV file.

For more details about the API endpoints and the request/response formats, refer to the Swagger documentation provided when you run the application.

ML model and research part RUS
---

### 1. Постановка Задачи:
* Определение цели проекта: построение модели для предсказания цены на автомобили.
* Использование набора данных с информацией о характеристиках автомобилей и их ценах.
### 2. Исследование Данных:
* Проведен анализ структуры данных и определение признаков, которые могут влиять на цену автомобиля. Выявлена кубическая зависимость цены от года, добавлен полином третьей степени от года в набор признаков. Из названия автомобиля получен бренд автомобиля. В совокупности, после этапа feature engineering удалось получить рост R2 на ~0.14.
### 3. Обработка пропущенных значений и выбросов в данных.
* Создан пайплайн для обработки категориальных и числовых признаков.
* Обработаны пропуски медианами распределения признаков полученных на обучающем наборе данных.
* Применен StandartScaler для масштабирования признаков.
* Применено кодирование категориальных признаков методом OneHot (с исключением первого столбца для каждого признака для избежания мультиколлинеарности).
### 4. Обучение Модели:
* Использована модель Ridge регрессии для предсказания цен.
* Подобраны оптимальные параметры модели с использованием GridSearchCV.
### 5. Развертывание API:
* Создан FastAPI веб-сервис для предсказания цен на автомобили.
* Реализованы эндпоинты для предсказаний по одному объекту, списку объектов и загруженному CSV файлу.
### 6. Результаты
#### Метрики модели:
* Бизнес метрика (доля предиктов, отличающихся от реальных цен на эти авто не более чем на 10% (в одну или другую сторону): 0.277
* r2_test 0.7908330728485418
* r2_train 0.7800161432569799 
* mse_test 120235122621.80447
* mse_train 63055807162.37864 
#### Параметры модели:
* 'alpha': 0.0001
### 7.API Сервис:
* API успешно развернут и доступен для использования.
* Проект структурирован и легко масштабируется для будущих улучшений.
  
**Скриншоты работы**
* **POST /predict_item** - Predict the price for a single car.
  
<img width="1609" alt="Screenshot 2023-11-30 at 00 39 07" src="https://github.com/alexander-tereshin/car_price_prediction/assets/107271811/4a79487e-62f4-4752-b776-924723773025">


* **POST /predict_items** - Predict prices for a list of cars.
  
<img width="1448" alt="Screenshot 2023-11-30 at 00 37 54" src="https://github.com/alexander-tereshin/car_price_prediction/assets/107271811/d846d1e4-5738-4386-b1a0-711f9d575425">

* **POST /predict_file** - Predict prices from a CSV file.
  
<img width="1114" alt="Screenshot 2023-11-30 at 16 12 35" src="https://github.com/alexander-tereshin/car_price_prediction/assets/107271811/6c17d13a-aabe-4c7b-8426-9c6c477312f8">


### 8. Наибольший Буст в качестве
* добавление полинома третьей степени от года
* добавление бренда автомобиля
### 9. Что сделать не вышло
* не хватило ресурсов поработать с крутящим моментом двигателя
### 10. Вывод
* Проект позволил разработать и развернуть модель предсказания цен на автомобили с использованием веб-сервиса. Несмотря на некоторые ограничения, удалось добиться хороших результатов и создать удобный механизм для будущего улучшения модели и сервиса.
* L1 регуляризация действительно зануляет веса у коллинеарных признаков (Тип коробки передач: либо Automatic либо Manual)
### Fun Fact :grin:
Наиболее положительно влияющий признак на прогноз модели:
* Brand_Lexus
  
Наиболее отрицательно влияющий признак на прогноз модели: 
* Brand_Peugeot
  
License
---
This project is licensed under the terms of the MIT license.
# car_price_prediction
