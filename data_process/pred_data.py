import os
import numpy as np
import pandas as pd
from data_process.tag_info import usa_whole_tags, transfer_tags
from data_process.data_path import monitoring_country_data_path, country_compose_data_path, predict_ldf_data_path, country_daily_data_path
from data_process.compose import source_select, get_in_clusters
from data_process.allocate import LdfInputCompose
from model.regressor import CaliforniaPredictLdfCompose, LimaPredictLdfCompose

class CaliforniaLdfInputCompose:
    def __init__(self, target_station_num: int, split_id: int, near_station_num: int, ldf_a: bool):
        self.target_station_num = target_station_num
        self.split_id = split_id
        self.near_station_num = near_station_num
        self.ldf_a = ldf_a
        self.pred_daily_path = country_daily_data_path["california"]
        self.compose_data_path = predict_ldf_data_path["california"]

    def _split_dataset(self, train_test_split_id: dict, source_type: str):
        monitoring_whole_data = pd.read_csv(monitoring_country_data_path["usa"])[usa_whole_tags]
        source_index, train_target_index = train_test_split_id['train_out_cluster'], train_test_split_id['train_in_cluster']
        source_data = monitoring_whole_data.loc[np.isin(monitoring_whole_data["cmaq_id"], source_index)]
        train_target_data = monitoring_whole_data.loc[np.isin(monitoring_whole_data["cmaq_id"], train_target_index)]
        source_data = source_select(source_data, source_type)
        return source_data, train_target_data
    
    def _stack_pred_data(self):
        target_data = []
        for date in range(1,366):
            target_date_data = pd.read_csv(f"{self.pred_daily_path}us-2011-satellite-day-{date}.csv")
            target_data.append(target_date_data)
        target_data = pd.concat(target_data)
        return target_data

    def _allocate_stations(self, source_dt: pd.DataFrame, train_target_dt: pd.DataFrame, pred_dt: pd.DataFrame, save_dir: str):
        station_allocate = LdfInputCompose(source_dt, train_target_dt, pred_dt, self.near_station_num, "california", self.ldf_a)
        station_allocate.allocate_pred_daily(True, save_dir)

    def save(self, source_type: str):
        train_test_data_id = get_in_clusters(country_compose_data_path["california"], self.target_station_num, 20)
        if self.ldf_a:
            save_dir = f"{self.compose_data_path}tl-cal-{self.target_station_num}/split-{self.split_id}/{source_type} nearest{self.near_station_num} ldf_a/"
        else:
            save_dir = f"{self.compose_data_path}tl-cal-{self.target_station_num}/split-{self.split_id}/{source_type} nearest{self.near_station_num}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        train_test_split_id = train_test_data_id[self.split_id]
        source_dt, train_target_dt = self._split_dataset(train_test_split_id, source_type)
        pred_dt = self._stack_pred_data()
        self._allocate_stations(source_dt, train_target_dt, pred_dt, save_dir)

class CaliforniaPredLdfSet:
    def __init__(self, target_station_num: int, source_type: str, near_station_num: int, feature_name: str, split_id: int, ldf_a: bool):
        self.target_station_num = target_station_num
        self.source_type = source_type
        self.near_station_num = near_station_num
        self.feature_name = feature_name
        self.split_id = split_id
        self.ldf_a = ldf_a
        self.pred_data_path = predict_ldf_data_path["california"]
        self.train_data_path = country_compose_data_path["california"]
        self.feature_compute = CaliforniaPredictLdfCompose(self.source_type, self.near_station_num, self.target_station_num, self.ldf_a)
        self._list_ldf_files()
        self.file_id = 0

    def _list_ldf_files(self):
        if self.ldf_a:
            self.ldf_file_dir = f"{self.pred_data_path}tl-cal-{self.target_station_num}/split-{self.split_id}/{self.source_type} nearest{self.near_station_num} ldf_a/"
            self.train_source_path = f"{self.train_data_path}tl-cal-{self.target_station_num}/split-{self.split_id}/{self.source_type} nearest{self.near_station_num} ldf_a dataset.npz"
        else:
            self.ldf_file_dir = f"{self.pred_data_path}tl-cal-{self.target_station_num}/split-{self.split_id}/{self.source_type} nearest{self.near_station_num}/"
            self.train_source_path = f"{self.train_data_path}tl-cal-{self.target_station_num}/split-{self.split_id}/{self.source_type} nearest{self.near_station_num} dataset.npz"
        self.ldf_files = np.sort([x for x in os.listdir(self.ldf_file_dir)])

    def _process_data(self):
        pred_data_path = self.ldf_file_dir + self.ldf_files[self.file_id]
        if self.feature_name == "NF":
            pred_feature = None
        elif self.feature_name == "LDF":
            pred_feature = self.feature_compute.compute_ldf(self.train_source_path, pred_data_path, self.split_id)
        pred_set = self.feature_compute.combine_input_feature(self.train_source_path, pred_data_path, pred_feature)
        pred_set["file"] = self.ldf_files[self.file_id]
        return pred_set

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.file_id >= len(self.ldf_files):
            raise StopIteration
        file_dataset = self._process_data()
        self.file_id += 1
        return file_dataset

class LimaLdfInputCompose:
    def __init__(self, near_station_num: int, ldf_a: bool):
        self.near_station_num = near_station_num
        self.ldf_a = ldf_a
        self.pred_daily_path = country_daily_data_path["lima"]
        self.compose_data_path = predict_ldf_data_path["lima"]

    def _date_str_convert(self, num: int):
        str_num = str(num)
        for d in range(3-len(str_num)):
            str_num = '0' + str_num
        return str_num

    def _read_dataset(self, source_type: str):
        monitoring_whole_data = pd.read_csv(monitoring_country_data_path["usa"])[transfer_tags["lima"]["source"] + ["rid"]]
        train_target_data = pd.read_csv(monitoring_country_data_path["lima"])[transfer_tags["lima"]["source"]]
        source_data = source_select(monitoring_whole_data, source_type)[transfer_tags["lima"]["source"]]
        return source_data, train_target_data
    
    def _stack_pred_data(self):
        target_data = []
        for date in range(1,366):
            str_date = self._date_str_convert(date)
            date_file = f"Lima_Predictions_2016_{str_date}.csv"
            if date_file not in os.listdir(self.pred_daily_path):
                continue
            target_date_data = pd.read_csv(f"{self.pred_daily_path}Lima_Predictions_2016_{str_date}.csv")
            target_data.append(target_date_data)
        target_data = pd.concat(target_data)
        return target_data

    def save_pred_daily(self, source_type: str):
        if self.ldf_a:
            save_dir = f"{self.compose_data_path}{source_type} nearest{self.near_station_num} ldf_a/"
        else:
            save_dir = f"{self.compose_data_path}{source_type} nearest{self.near_station_num}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        source_dt, train_target_dt = self._read_dataset(source_type)
        pred_dt = self._stack_pred_data()
        station_allocate = LdfInputCompose(source_dt, train_target_dt, pred_dt, self.near_station_num, "lima", self.ldf_a)
        station_allocate.allocate_pred_daily(False, save_dir)

    def save_monitor(self, source_type: str):
        if self.ldf_a:
            file_name = f"{self.compose_data_path}monitor {source_type} nearest{self.near_station_num} ldf_a dataset.npz"
        else:
            file_name = f"{self.compose_data_path}monitor {source_type} nearest{self.near_station_num} dataset.npz"
        source_dt, train_target_dt = self._read_dataset(source_type)
        station_allocate = LdfInputCompose(source_dt, train_target_dt, train_target_dt, self.near_station_num, "lima", self.ldf_a)
        allocate_input, allocate_label = station_allocate.allocate_all(False)
        np.savez(file_name,
            source_input = allocate_input["source"], 
            train_target_input = allocate_input["train_target"],
            valid_input = allocate_input["valid"],
            source_label = allocate_label["source"], 
            train_target_label = allocate_label["train_target"],
            valid_label = allocate_label["valid"],)

class LimaPredLdfSet:
    def __init__(self, source_type: str, near_station_num: int, feature_name: str, ldf_a: bool):
        self.source_type = source_type
        self.near_station_num = near_station_num
        self.feature_name = feature_name
        self.ldf_a = ldf_a
        self.pred_data_path = predict_ldf_data_path["lima"]
        self.train_data_path = country_compose_data_path["lima"]
        self.feature_compute = LimaPredictLdfCompose(self.source_type, self.near_station_num, self.ldf_a)
        self._list_ldf_files()
        self.file_id = 0

    def _list_ldf_files(self):
        if self.ldf_a:
            self.ldf_file_dir = f"{self.pred_data_path}{self.source_type} nearest{self.near_station_num} ldf_a/"
        else:
            self.ldf_file_dir = f"{self.pred_data_path}{self.source_type} nearest{self.near_station_num}/"
        self.ldf_files = np.sort([x for x in os.listdir(self.ldf_file_dir)])

    def _process_data(self):
        pred_data_path = self.ldf_file_dir + self.ldf_files[self.file_id]
        if self.feature_name == "NF":
            pred_feature = None
        elif self.feature_name == "LDF":
            pred_feature = self.feature_compute.compute_pred_ldf(pred_data_path, False)
        pred_set = self.feature_compute.combine_pred_input_feature(pred_data_path, pred_feature)
        pred_set["file"] = self.ldf_files[self.file_id]
        return pred_set

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.file_id >= len(self.ldf_files):
            raise StopIteration
        file_dataset = self._process_data()
        self.file_id += 1
        return file_dataset
