import os
from data_process.data_path import country_compose_data_path, predict_ldf_data_path
from data_process.compose import LdfInputData
from model.autoencoder import TrainLdfModel

class LdfComposer:
    def __init__(self, transfer_name: str, source_type: str,  nearest_station_num: int, ldf_a: bool):
        self.input_data = LdfInputData(transfer_name, source_type, nearest_station_num, ldf_a)
        self.train_model = TrainLdfModel(1, self.input_data.target_dim, self.input_data.input_shape)

    def train(self, data_path: str, model_path: str):
        train_loader, _ = self.input_data.convert_loader(True, data_path)
        self.train_model.train(model_path, train_loader, 30)

    def encode(self, data_path: str, model_path: str):
        source_dt, train_target_dt, valid_dt = self.input_data.convert_loader(False, data_path)
        source_encode = self.train_model.encode(model_path, source_dt)
        train_target_encode = self.train_model.encode(model_path, train_target_dt)
        valid_encode = self.train_model.encode(model_path, valid_dt)
        return source_encode, train_target_encode, valid_encode
    
    def pred_encode(self, train_source_file: str, pred_file: str, model_path: str):
        pred_dt = self.input_data.convert_pred_loader(train_source_file, pred_file)
        pred_encode = self.train_model.encode(model_path, pred_dt)
        return pred_encode

class CaliforniaSplitLdfCompose:
    def __init__(self, source_type: str,  nearest_station_num: int, target_station_num: int, ldf_a: bool):
        self.source_type = source_type
        self.nearest_station_num = nearest_station_num
        self.target_station_num = target_station_num
        self.ldf_a = ldf_a
        self.compose_data_path = country_compose_data_path["california"]
        self.ldf_composer = LdfComposer("california", source_type, nearest_station_num, ldf_a)

    def compute_ldf(self, split_id: int, train: bool):
        model_dir = f"trained models/ldf composer/tl-cal-{self.target_station_num}/split-{split_id}/{self.source_type} nearest{self.nearest_station_num}/"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if self.ldf_a:
            data_path = f"{self.compose_data_path}tl-cal-{self.target_station_num}/split-{split_id}/{self.source_type} nearest{self.nearest_station_num} ldf_a dataset.npz"
            model_path = model_dir + "ldf-a_model"
        else:
            data_path = f"{self.compose_data_path}tl-cal-{self.target_station_num}/split-{split_id}/{self.source_type} nearest{self.nearest_station_num} dataset.npz"
            model_path = model_dir + "vanilla_model"
        if train:
            self.ldf_composer.train(data_path, model_path)
        source_encode, train_target_encode, valid_encode = self.ldf_composer.encode(data_path, model_path)
        return source_encode, train_target_encode, valid_encode
    
    def combine_input_feature(self, split_id: int, all_features: dict):
        data_path = f"{self.compose_data_path}tl-cal-{self.target_station_num}/split-{split_id}/{self.source_type} nearest{self.nearest_station_num} dataset.npz"
        input_data = LdfInputData("california", self.source_type, self.nearest_station_num, False)
        source_set, train_target_set, valid_set = input_data.compose_regress_input(data_path, all_features)
        return source_set, train_target_set, valid_set

class CaliforniaPredictLdfCompose:
    def __init__(self, source_type: str,  nearest_station_num: int, target_station_num: int, ldf_a: bool):
        self.source_type = source_type
        self.nearest_station_num = nearest_station_num
        self.target_station_num = target_station_num
        self.ldf_a = ldf_a
        self.compose_data_path = country_compose_data_path["california"]
        self.ldf_composer = LdfComposer("california", source_type, nearest_station_num, ldf_a)

    def compute_ldf(self, train_source_path: str, data_path: str, split_id: int):
        model_dir = f"trained models/ldf composer/tl-cal-{self.target_station_num}/split-{split_id}/{self.source_type} nearest{self.nearest_station_num}/"
        if self.ldf_a:
            model_path = model_dir + "ldf-a_model"
        else:
            model_path = model_dir + "vanilla_model"
        pred_encode = self.ldf_composer.pred_encode(train_source_path, data_path, model_path)
        return pred_encode
    
    def combine_input_feature(self, train_source_path: str, data_path: str, all_features: dict):
        input_data = LdfInputData("california", self.source_type, self.nearest_station_num, False)
        pred_set = input_data.compose_pred_regress_input(train_source_path, data_path, all_features)
        return pred_set

class LimaPredictLdfCompose:
    def __init__(self, source_type: str,  nearest_station_num: int, ldf_a: bool):
        self.source_type = source_type
        self.nearest_station_num = nearest_station_num
        self.ldf_a = ldf_a
        self.compose_data_path = predict_ldf_data_path["lima"]
        self.ldf_composer = LdfComposer("lima", source_type, nearest_station_num, ldf_a)
        self.train_source_path = f"{self.compose_data_path}monitor {source_type} nearest{self.nearest_station_num} dataset.npz"
        if self.ldf_a:
            self.train_source_ldf_path = f"{self.compose_data_path}monitor {source_type} nearest{self.nearest_station_num} ldf_a dataset.npz"
        else:
            self.train_source_ldf_path = self.train_source_path

    def compute_train_ldf(self, train: bool):
        model_dir = f"trained models/lima/ldf composer/{self.source_type} predict "
        if self.ldf_a:
            model_path = model_dir + "ldf-a_model"
        else:
            model_path = model_dir + "vanilla_model"
        if train:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            self.ldf_composer.train(self.train_source_ldf_path, model_path)
        source_encode, train_target_encode, valid_encode = self.ldf_composer.encode(self.train_source_ldf_path, model_path)
        return source_encode, train_target_encode, valid_encode

    def combine_train_input_feature(self, all_features: dict):
        input_data = LdfInputData("lima", self.source_type, self.nearest_station_num, False)
        source_set, train_target_set, valid_set = input_data.compose_regress_input(self.train_source_path, all_features)
        return source_set, train_target_set, valid_set

    def compute_pred_ldf(self, pred_data_path: str):
        model_dir = f"trained models/lima/ldf composer/{self.source_type} predict "
        if self.ldf_a:
            model_path = model_dir + "ldf-a_model"
        else:
            model_path = model_dir + "vanilla_model"
        pred_encode = self.ldf_composer.pred_encode(self.train_source_ldf_path, pred_data_path, model_path)
        return pred_encode

    def combine_pred_input_feature(self, pred_data_path: str, all_features: dict):
        input_data = LdfInputData("lima", self.source_type, self.nearest_station_num, False)
        pred_set = input_data.compose_pred_regress_input(self.train_source_path, pred_data_path, all_features)
        return pred_set
