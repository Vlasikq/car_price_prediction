import pickle
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


class CarPricePredictorPreprocessor:
    def __init__(self, models_folder):
        # Загрузка сохранённых моделей и объектов
        self.na_imputer = self.load_pickle(models_folder, filename='imputer.pkl')
        self.normalizer = self.load_pickle(models_folder, filename='scaler.pkl')
        self.ohe = self.load_pickle(models_folder, filename='ohe.pkl')
        self.ridge_regressor = self.load_pickle(models_folder, filename='ridge_regressor.pkl')

    @staticmethod
    def load_pickle(folder, filename):
        # Метод загрузки файла pickle
        with open(folder / filename, 'rb') as file:
            return pickle.load(file)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Обработка данных
        for col in ['mileage', 'engine', 'max_power']:
            df[col] = df[col].apply(lambda x: float(''.join(c for c in str(x) if c.isdigit() or c == '.')) if pd.notna(x) else None)

        cat_features = ['fuel', 'seller_type', 'transmission', 'owner']
        df_cat = df[cat_features].copy()
        df_cat_encoded = pd.DataFrame(self.ohe.transform(df_cat), columns=self.ohe.get_feature_names_out())

        df_num = df.select_dtypes(include=['int64', 'float64']).drop(columns=['selling_price'], errors='ignore')
        df_num_filled = pd.DataFrame(self.na_imputer.transform(df_num), columns=df_num.columns)
        df_num_filled[['engine', 'seats']] = df_num_filled[['engine', 'seats']].astype(int)

        df_final = pd.concat([df_num_filled, df_cat_encoded], axis=1)
        return pd.DataFrame(self.normalizer.transform(df_final), columns=df_final.columns)
