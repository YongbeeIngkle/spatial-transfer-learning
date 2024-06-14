import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from pykrige.ok import OrdinaryKriging
from data_process.tag_info import ldf_input_tags, input_tags
from model.regressor import SplitTargetRead

class TrainTest:
    def __init__(self, model_name: str, t_num: int):
        self.model_name = model_name
        self.data_reader = SplitTargetRead(12, t_num)
        self._define_regressor()

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
            self.model = self._define_rf()
        elif self.model_name == "GBR":
            self.model = self._define_gbr()

    def _train_model(self, target_x, target_y):
        self.model.fit(target_x, target_y)

    def train(self, train_target_dataset: dict):
        train_target_input = train_target_dataset["input"]
        train_target_label = train_target_dataset["label"]
        self._train_model(train_target_input, train_target_label)

    def predict(self, split_id: int):
        train_target_dataset, pred_dataset = self.data_reader.read_dataset(True, split_id)
        self.train(train_target_dataset)
        input_dt = pred_dataset["input"]
        label_dt = pred_dataset["label"]
        pred_val = self.model.predict(input_dt)
        mse_val = mean_squared_error(np.array(label_dt), pred_val)
        print(f"MSE value: {mse_val}")
        return pred_val, np.array(label_dt)

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

class Kriger:
    def __init__(self, t_num):
        self.data_reader = SplitTargetRead(12, t_num)
        
    def _train_model(self, target_x, target_y):
        self.model.fit(target_x, target_y)

    def _train(self, train_data: pd.DataFrame):
        self.model = OrdinaryKriging(train_data["cmaq_x"], train_data["cmaq_y"], train_data["pm25_value"],
                                     variogram_model="linear", verbose=False, enable_plotting=False)
        
    def _get_one_pred(self, test_data: pd.DataFrame):
        z, _ = self.model.execute("points", test_data["cmaq_x"], test_data["cmaq_y"])
        pred_vals = z.data
        return pred_vals

    def predict(self, split_id):
        train_target_dataset, pred_dataset = self.data_reader.read_dataset(False, split_id)
        train_target_dt = pd.DataFrame(train_target_dataset["input"], columns=ldf_input_tags)
        train_target_dt["pm25_value"] = train_target_dataset["label"]
        pred_dt = pd.DataFrame(pred_dataset["input"], columns=ldf_input_tags)
        pred_dt["pm25_value"] = pred_dataset["label"]

        unique_dates = np.unique(train_target_dt["day"])
        pred, label = [], []
        for date in unique_dates:
            try:
                date_train_target_dt = train_target_dt[train_target_dt["day"]==date]
                date_pred_dt = pred_dt[pred_dt["day"]==date]
                self._train(date_train_target_dt)
                date_pred = self._get_one_pred(date_pred_dt)
                pred.append(date_pred)
                label.append(np.array(date_pred_dt["pm25_value"]))
            except:
                continue
        pred = np.hstack(pred)
        label = np.hstack(label)
        mse_val = mean_squared_error(label, pred)
        print(f"MSE value: {mse_val}")
        return pred, label

class CountryTrainPred:
    def __init__(self, model_name: str, input_normalize: bool):
        self.model_name = model_name
        self.input_normalize = input_normalize
        self._define_regressor()

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
            self.model = self._define_rf()
        elif self.model_name == "GBR":
            self.model = self._define_gbr()

    def _normalize_input(self, dataset: pd.DataFrame, train: bool):
        if train:
            self.mean = dataset.mean()
            self.std = dataset.std()
            self.std.loc[self.std==0] = 1
        return (dataset - self.mean) / self.std

    def train(self, train_data: pd.DataFrame):
        train_input = train_data[input_tags]
        if self.input_normalize:
            train_input = self._normalize_input(train_input, True)
        train_label = train_data["pm25_value"]
        self.model.fit(train_input, train_label)

    def predict(self, pred_data: pd.DataFrame):
        pred_input = pred_data[input_tags]
        if self.input_normalize:
            pred_input = self._normalize_input(pred_input, False)
        pred_val = self.model.predict(pred_input)
        return pred_val

