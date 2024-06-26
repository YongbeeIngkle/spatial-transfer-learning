import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

class TrainTestTransferData:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _define_rf(self):
        params = dict({
            "n_estimators": 400,
            "max_leaf_nodes": 4,
            "max_depth": None,
            "random_state": 2,
            "min_samples_split": 5  
        })
        model = RandomForestRegressor(**params)
        return model

    def _define_gbr(self):
        params = dict({
            "n_estimators": 400,
            "max_leaf_nodes": 4,
            "max_depth": None,
            "random_state": 2,
            "min_samples_split": 5,
            "learning_rate": 0.1,
            "subsample": 0.5
        })
        model = GradientBoostingRegressor(**params)
        return model
    
    def _define_regressor(self):
        if self.model_name == "RF":
            model = self._define_rf()
        elif self.model_name == "GBR":
            model = self._define_gbr()
        return model

    def _train_model(self, source_x, source_y, target_x, target_y):
        model = self._define_regressor()
        input_data = np.vstack([source_x, target_x])
        label_data = np.hstack([source_y, target_y])
        model.fit(input_data, label_data)
        return model

    def train(self, source_dataset: dict, train_target_dataset: dict):
        source_input = source_dataset["input"]
        source_label = source_dataset["label"]
        train_target_input = train_target_dataset["input"]
        train_target_label = train_target_dataset["label"]
        self.model = self._train_model(source_input, source_label, train_target_input, train_target_label)

    def predict(self, pred_dataset: dict, return_label=True):
        input_dt = pred_dataset["input"]
        pred_val = self.model.predict(input_dt)
        if return_label:
            label_dt = pred_dataset["label"]
            mse_val = mean_squared_error(np.array(label_dt), pred_val)
            print(f"MSE value: {mse_val}")
            return pred_val, np.array(label_dt)
        else:
            return pred_val
