# Car Price Prediction API

## Обзор проекта
Этот проект направлен на создание модели для предсказания стоимости автомобилей на основе их характеристик. В ходе работы выполнен полный цикл подготовки данных, обучения модели и оценки её качества с использованием различных подходов. А также частично выполнина реализация FastAPI сервиса
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
# Car Price Prediction Project

## Overview
Этот проект направлен на создание модели для предсказания стоимости автомобилей на основе их характеристик. В ходе работы выполнен полный цикл подготовки данных, обучения модели и оценки её качества с использованием различных подходов.

---

## Результаты работы

### Предобработка данных и EDA
1. **Анализ данных**:
   - Проведён первичный анализ данных: определены типы полей, выявлены пропуски, составлен их список.
   - Использована библиотека `ydata_profiling` для создания отчёта, который сохранён в файл `profile_report.html`.

2. **Обработка признаков**:
   - Поля `mileage`, `engine`, `max_power` и  очищены от единиц измерения и приведены к числовым типам.
   - Для обработки сложного признака `torque` не стал далеко уходить и удалил его

3. **Заполнение пропусков**:
   - Пропуски в числовых полях заполнены медианными значениями, рассчитанными по тренировочному набору данных.

4. **Удаление дубликатов**:
   - Из тренировочного набора удалены записи с одинаковыми признаками, обновлены индексы строк.

5. **Анализ распределений и корреляций**:
   - Построены графики распределений (`pairplot`) для тренировочного и тестового наборов данных.
   - Использованы тепловые карты (`heatmap`) для визуализации корреляций признаков:
     - Наибольшая зависимость с таргетом (`selling_price`) наблюдается у признаков `max_power` и `engine`.
     - Выявлена зависимомть между признаками `engine`, `mileage`, `max_power` и `seats`.

6. **Сравнение распределений тестового и тренировочного наборов**:
   - Выявлено схожее распределение большинства признаков, за исключением года выпуска


#### Параметры модели:
### Модели на вещественных признаках
1. **Линейная регрессия**:
   - Без стандартизации:  
    `r2_train` 0.5932097784368064
    `mse_train` 116601673169.11029
    `r2_test` 0.5946576472666452
    `mse_test` 233002359160.8032
   - Со стандартизацией:  
     - Качество модели практически не изменилось
    `r2_train` 0.5932097784368073
    `mse_train` 116601673169.11008
    `r2_test` 0.5946576472666552
    `mse_test` 233002359160.7975

2. **Lasso-регрессия**:
    `r2_test` 0.5940653519598107
    `r2_train` 0.5932010464312698
    `mse_test` 233342827416.54224
    `mse_train` 116604176096.66785
    `best_params` {'alpha': 506}
    `Грид-сёрчу пришлось обучать 10 моделей`
    `CPU times`: total:` 0 ns
    `Wall time`: 31.6 ms
   - Занулённые признаки: `km_driven`, `mileage`, `engine`, `seats`.

3. **ElasticNet-регрессия**:
    `r2_test` 0.5924660913050379
    `r2_train` 0.593154683361061
    `Грид-сёрчу пришлось обучать 70 моделей`
    `best_params` {'alpha': 506}
    `mse_test` 234262128108.81476
    `mse_train` 116617465530.07367
    `Зануленные признаки`: 0
    `CPU times`: total: 703 ms
    `Wall time`: 3.3 s 

### Модели с категориальными признаками
1. **Обработка категориальных признаков**:
   - Поле `name` удалил
   - Категориальные признаки закодированы методом OneHot Encoding.

2. **Гребневая регрессия (Ridge)**:
    `best_alpha_ridge`: 10
    `best_r2_ridge`: 0.622575871803245
    `test_r2_ridge`: 0.6456821792653527

---

### Бизнес-метрика

`Бизнес метрика (y_test)` - 0.245
- **Лучшая модель**: Ridge-регрессия (`alpha = 10`).  
 

### 7.API Сервис:
* API: много попыток развернут всегда какие-то проблемы, один раз был успешен
  


### 9. Что сделать не вышло
По работе с данными не хватило времени поработать с крутящим моментом двигателя и стобцом name с добавление бренда автомобиля. Также не успел с анализом аномалий в данных 
От себя: Возникло много трудностей с написание сервиса.
Для мнея это первый опыт и поэтому не хватило времени и усвоить всю информацию. Также понимаю что с даннными еще можно плотно поработать для улучшения качесвта предсказания. Получил ценный опыт в построение модель предсказания цен на автомобили с использованием сервиса


### 10. Вывод
Научился рабоать с моделями для предсказания стоимости автомобилей и часичто с развертыванием сервиса Fast API.
Проект позволил разработать и развернуть модель предсказания цен на автомобили с использованием веб-сервиса. Несмотря на некоторые ограничения, удалось добиться результатов и создать удобный механизм для будущего улучшения модели и сервиса.


