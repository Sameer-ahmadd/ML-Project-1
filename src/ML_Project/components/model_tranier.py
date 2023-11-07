import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from utils import save_object,evaluate_models
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor



@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            model_report:dict = evaluate_models(x_train= x_train, x_test = x_test ,y_train = y_train,
            y_test = y_test,models = models)
            

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException('No best Model Found')
            
            logging.info("Best model found on both training and test data")

            save_object(
                file_path= self.model_trainer_config.trained_model_path,
                obj = best_model
            )
            predicted = best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
     
        except Exception as e:
            raise CustomException(e,sys)


