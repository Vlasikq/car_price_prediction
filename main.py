import pathlib
import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List

from preprocessing import CarPricePredictorPreprocessor

app = FastAPI()


class Item(BaseModel):
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: int
    max_power: float
    seats: int


class Items(BaseModel):
    objects: List[Item]

models_folder = pathlib.Path('./models')
preprocessor = CarPricePredictorPreprocessor(models_folder)


def predict_price(item: Item) -> float:
    """
    Предсказание цены для одного объекта.
    """
    processed_df = preprocessor.preprocess_data(pd.DataFrame([item.dict()]))
    return preprocessor.ridge_regressor.predict(processed_df)[0]


def predict_prices(items: Items) -> List[float]:
    """
    Предсказание цен для списка объектов.
    """
    df = pd.DataFrame([item.dict() for item in items.objects])
    processed_df = preprocessor.preprocess_data(df)
    return preprocessor.ridge_regressor.predict(processed_df).tolist()


@app.get(path="/")
async def root():
    """
    Корневой эндпоинт с приветственным сообщением.
    """
    return {"message": "Hello, User! This is The Car Price Prediction Web Service!"}


@app.post("/predict_item")
async def predict_item(item: Item) -> float:
    """
    Эндпоинт для предсказания цены одного объекта.
    """
    try:
        return predict_price(item)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_items")
async def predict_items(items: Items) -> List[float]:
    """
    Эндпоинт для предсказания цен нескольких объектов.
    """
    try:
        return predict_prices(items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/predict_file")
async def predict_file(file: UploadFile):
    try:
        df = pd.read_csv(file.file)
        print("Входной файл прочитан, данные:", df.head())

        processed_df = preprocessor.preprocess_data(df)
        print("Обработанные данные:", processed_df.head())

        predictions = preprocessor.ridge_regressor.predict(processed_df).tolist()
        df["predicted_price"] = predictions
        output_path = "output_predictions.csv"
        df.to_csv(output_path, index=False)

        return {"file_path": output_path}
    except Exception as e:
        print("Ошибка:", e)
        raise HTTPException(status_code=500, detail=str(e))