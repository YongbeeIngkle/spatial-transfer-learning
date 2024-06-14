import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import tensorflow as tf
from mapie.regression import MapieRegressor
from mapie.metrics import regression_coverage_score_v2
from adapt.instance_based import TrAdaBoostR2, KMM, KLIEP, RULSIF, NearestNeighborsWeighting
from adapt.feature_based import DeepCORAL, DANN
from adapt.parameter_based import FineTuning, RegularTransferNN

algorithm_class = {
    "gbr": ["Kmm", "Kliep", "Rulsif", "Nnw"],
    "nn_feature": ["DeepCoral", "Dann"],
    "nn_parameter": ["FineTuning", "RegularTransferNN"]
}

class GbrTrainTest:
    def __init__(self, model_name: str):
        self.model_name = model_name

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
    
    def _compute_adapt(self, source_x, target_x):
        if self.model_name == "Kmm":
            weight_model = KMM(DecisionTreeRegressor(max_depth = 6), Xt=target_x, kernel="rbf", gamma=0.1, verbose=0, random_state=0)
        elif self.model_name == "Kliep":
            weight_model = KLIEP(DecisionTreeRegressor(max_depth = 6), Xt=target_x, kernel="poly", gamma=0.1, verbose=0, random_state=0)
        elif self.model_name == "Rulsif":
            weight_model = RULSIF(DecisionTreeRegressor(max_depth = 6), kernel="rbf", alpha=0.1, lambdas=[0.1, 1., 10.], gamma=[0.1, 1., 10.], Xt = target_x, random_state=2)
        elif self.model_name == "Nnw":
            weight_model = NearestNeighborsWeighting(DecisionTreeRegressor(max_depth = 6), Xt=target_x, n_neighbors=6, random_state=0)
        weights = weight_model.fit_weights(source_x, target_x)
        weights = np.asarray(weights)
        weights = weights.reshape((weights.size, 1))
        source_adapt_x = source_x*weights
        return source_adapt_x

    def _train_model(self, source_x, source_y, target_x, target_y):
        model = self._define_gbr()
        source_adapt_x = self._compute_adapt(source_x, target_x)
        if type(source_adapt_x) == pd.DataFrame:
            input_data = pd.concat([source_adapt_x, target_x])
            label_data = pd.concat([source_y, target_y])
        else:
            input_data = np.vstack([source_adapt_x, target_x])
            label_data = np.hstack([source_y, target_y])
        model.fit(input_data, label_data)
        return model

    def train(self, source_dataset: dict, train_target_dataset: dict):
        self.all_models = {}
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
        
    def mapie(self, source_dataset: dict, train_target_dataset: dict, pred_dataset: dict):
        X_train = np.vstack([source_dataset["input"], train_target_dataset["input"]])
        y_train = np.hstack([source_dataset["label"], train_target_dataset["label"]])
        X_test = pred_dataset["input"]
        y_test = pred_dataset["label"]
        mapie_regressor = MapieRegressor(self.model)
        mapie_regressor.fit(X_train, y_train)

        alpha = [0.05, 0.32]
        y_pred, y_pis = mapie_regressor.predict(X_test, alpha=alpha)
        coverage_scores = regression_coverage_score_v2(y_test, y_pis)
        return coverage_scores

class NnFeatureTrainTest:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._define_fnn()

    def _define_fnn(self):
        self.encoder_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
            ])
        self.task_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
            ])

    def _train_model(self, source_x, source_y, target_x):
        if self.model_name == "DeepCoral":
            self.model = DeepCORAL(encoder=self.encoder_model, task=self.task_model, Xt=target_x, lambda_=0.1, metrics=["mse"], random_state=0)
        elif self.model_name == "Dann":
            self.model = DANN(encoder=self.encoder_model, task=self.task_model, Xt=target_x, lambda_=0.1, metrics=["mse"], random_state=0)
        self.model.fit(source_x, source_y.reshape(-1,1), epochs=10, verbose=1)

    def train(self, source_dataset: dict, train_target_dataset: dict):
        source_input = source_dataset["input"]
        source_label = source_dataset["label"]
        train_target_input = train_target_dataset["input"]
        self._train_model(source_input, source_label, train_target_input)

    def predict(self, pred_dataset: dict):
        input_dt = pred_dataset["input"]
        label_dt = pred_dataset["label"]
        pred_val = self.model.predict(input_dt)
        mse_val = mean_squared_error(np.array(label_dt), pred_val)
        print(f"MSE value: {mse_val}")
        return pred_val, np.array(label_dt)

class NnParameterTrainTest:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._define_fnn()

    def _define_fnn(self):
        self.encoder_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
            ])
        self.task_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
            ])

    def _train_model(self, source_x, source_y, target_x, target_y):
        if self.model_name == "FineTuning":
            src_model = FineTuning(encoder=self.encoder_model, loss="mse", pretrain=False, random_state=0)
            src_model.fit(source_x, source_y.reshape(-1,1), epochs=10, verbose=1)
            self.model = FineTuning(encoder=src_model.encoder_, task=self.task_model, loss="mse", pretrain=True, pretrain__epochs=10, random_state=0)
            self.model.fit(target_x, target_y.reshape(-1,1), epochs=50, verbose=1)
        elif self.model_name == "RegularTransferNN":
            src_model = RegularTransferNN(task=self.encoder_model, loss="mse", random_state=0)
            src_model.fit(source_x, source_y.reshape(-1,1), epochs=10, verbose=1)
            self.model = RegularTransferNN(task=src_model.task_, loss="mse", random_state=0)
            self.model.fit(target_x, target_y.reshape(-1,1), epochs=50, verbose=1)

    def train(self, source_dataset: dict, train_target_dataset: dict):
        source_input = source_dataset["input"]
        source_label = source_dataset["label"]
        train_target_input = train_target_dataset["input"]
        train_target_label = train_target_dataset["label"]
        self._train_model(source_input, source_label, train_target_input, train_target_label)

    def predict(self, pred_dataset: dict):
        input_dt = pred_dataset["input"]
        label_dt = pred_dataset["label"]
        pred_val = self.model.predict(input_dt).flatten()
        mse_val = mean_squared_error(np.array(label_dt), pred_val)
        print(f"MSE value: {mse_val}")
        return pred_val, np.array(label_dt)

class AggregateTrainTest:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _define_model(self):
        if self.model_name == "GBR":
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

    def _train_model(self, source_x, source_y, target_x, target_y):
        model = self._define_model()
        input_data = np.vstack([source_x, target_x])
        label_data = np.hstack([source_y, target_y])
        model.fit(input_data, label_data)
        return model

    def train(self, source_dataset: dict, train_target_dataset: dict):
        self.all_models = {}
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
