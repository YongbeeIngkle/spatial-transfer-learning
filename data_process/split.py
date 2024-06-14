import os
import pandas as pd
import numpy as np
from data_process.tag_info import usa_whole_tags, transfer_tags
from data_process.data_path import monitoring_data_path, country_compose_data_path, monitoring_country_data_path
from data_process.compose import source_select, get_in_clusters
from data_process.allocate import LdfInputCompose

class CaliforniaSourcetargetId:
    def __init__(self, cluster_id: int, target_train_numbers: list):
        """
        Split the source and target stations.

        cluster_id: target area's cluster(rid) id.
        target_train_numbers: number of train-target stations
        """
        self.cluster_id = cluster_id
        self.target_train_numbers = target_train_numbers
        self.compose_data_path = country_compose_data_path["california"]

    def _source_target_split(self, whole_data: pd.DataFrame):
        """
        Split the source-target based on the rid.
        """
        target_data = whole_data.loc[whole_data["rid"] == self.cluster_id]
        source_data = whole_data.loc[whole_data["rid"] != self.cluster_id]
        target_cmaqs, target_count = np.unique(target_data[["cmaq_x", "cmaq_y"]], axis=0, return_counts=True)
        source_cmaqs = np.unique(source_data[["cmaq_x", "cmaq_y"]], axis=0)
        cmaq_set = {"source": source_cmaqs, "target": target_cmaqs}
        return cmaq_set, target_count
        
    def _read_input_label(self):
        """
        Read the input-label data and split into source and target cmaqs.
        """
        monitoring_whole_data = pd.read_csv(monitoring_data_path)[usa_whole_tags]
        source_target_set, target_count = self._source_target_split(monitoring_whole_data)
        return source_target_set, target_count
    
    def _split_data_coord(self, cmaq_coords: np.ndarray, train_num: int, split_num: int):
        """
        Split coordinate data into train-test dictionary.
        """
        np.random.seed(1000)
        cmaq_ids = np.arange(len(cmaq_coords))
        set_dataset = {}
        for set_id in range(split_num):
            train_cmaq_ids = np.random.choice(cmaq_ids, train_num, replace=False)
            test_cmaq_ids = np.array([i for i in cmaq_ids if i not in train_cmaq_ids])
            set_dataset[set_id] = {"train": cmaq_coords[train_cmaq_ids], "test": cmaq_coords[test_cmaq_ids]}
        return set_dataset
    
    def _save_source_cmaqs(self, source_coord):
        """
        Save the source cmaq_id's.
        """
        save_file = f"{self.compose_data_path}source_cmaq.npy"
        np.save(save_file, source_coord)
    
    def _save_target_cmaqs(self, target_coord_set, train_num):
        """
        Save the train-target and validation cmaq_id's.
        """
        for set_id in target_coord_set.keys():
            set_data = target_coord_set[set_id]
            train_cmaq = np.array(set_data["train"])
            test_cmaq = np.array(set_data["test"])
            save_dir = f'{self.compose_data_path}tl-cal-{train_num}/split-{set_id}/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_file = f"{save_dir}target_cmaq.npz"
            np.savez(save_file, train=train_cmaq, test=test_cmaq)
    
    def save(self, split_num: int):
        """
        Save the source, train-target, validation station set for each split.

        split_num: number of splits
        """
        source_target_set, target_count = self._read_input_label()
        self._save_source_cmaqs(source_target_set["source"])
        for train_num in self.target_train_numbers:
            ### Choose the train-target stations having measurement more than 300 of 365 days.
            choice_targets = source_target_set["target"][target_count>300]
            train_test_coord = self._split_data_coord(choice_targets, train_num, split_num)
            self._save_target_cmaqs(train_test_coord, train_num)

class LimaSourceTargetId:
    def __init__(self, target_train_numbers: list):
        """
        Split the source and target stations.

        cluster_id: target area's cluster(rid) id.
        target_train_numbers: number of train-target stations
        """
        self.target_train_numbers = target_train_numbers
        self.lima_data_path = country_compose_data_path["lima"]

    def _source_target_split(self, usa_data: pd.DataFrame, country_data: pd.DataFrame):
        """
        Split the source-target based on the rid.
        """
        target_coords, target_count = np.unique(country_data[["lat", "lon"]], axis=0, return_counts=True)
        source_coords = np.unique(usa_data[["lat", "lon"]], axis=0)
        coord_set = {"source": source_coords, "target": target_coords}
        return coord_set, target_count

    def _read_input_label(self):
        """
        Read the input-label data and split into source and target cmaqs.
        """
        monitoring_usa_data = pd.read_csv(monitoring_country_data_path["usa"])[transfer_tags["lima"]["source"]]
        monitoring_lima_data = pd.read_csv(monitoring_country_data_path["lima"])[transfer_tags["lima"]["source"]]
        source_target_set, target_count = self._source_target_split(monitoring_usa_data, monitoring_lima_data)
        return source_target_set, target_count
    
    def _split_data_coord(self, coords: np.ndarray, train_num: int, split_num: int):
        """
        Split coordinate data into train-test dictionary.
        """
        np.random.seed(1000)
        coord_ids = np.arange(len(coords))
        set_dataset = {}
        for set_id in range(split_num):
            train_coords = np.random.choice(coord_ids, train_num, replace=False)
            test_coords = np.array([i for i in coord_ids if i not in train_coords])
            set_dataset[set_id] = {"train": coords[train_coords], "test": coords[test_coords]}
        return set_dataset
    
    def _save_source_cmaqs(self, source_coord):
        """
        Save the source cmaq_id's.
        """
        save_file = f"{self.lima_data_path}source_cmaq.npy"
        np.save(save_file, source_coord)
    
    def _save_target_cmaqs(self, target_coord_set, train_num):
        """
        Save the train-target and validation cmaq_id's.
        """
        for set_id in target_coord_set.keys():
            set_data = target_coord_set[set_id]
            train_cmaq = np.array(set_data["train"])
            test_cmaq = np.array(set_data["test"])
            save_dir = f'{self.lima_data_path}tl-cal-{train_num}/split-{set_id}/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_file = f"{save_dir}target_cmaq.npz"
            np.savez(save_file, train=train_cmaq, test=test_cmaq)
    
    def save(self, split_num: int, count_lower: int):
        """
        Save the source, train-target, validation station set for each split.

        split_num: number of splits
        """
        source_target_set, target_count = self._read_input_label()
        self._save_source_cmaqs(source_target_set["source"])
        for train_num in self.target_train_numbers:
            ### Choose the train-target stations having measurement more than 300 of 365 days.
            choice_targets = source_target_set["target"][target_count>count_lower]
            train_test_coord = self._split_data_coord(choice_targets, train_num, split_num)
            self._save_target_cmaqs(train_test_coord, train_num)

class CaliforniaSplitLdfInputCompose:
    def __init__(self, target_station_num: int, station_num: int, ldf_a: bool):
        self.target_station_num = target_station_num
        self.station_num = station_num
        self.ldf_a = ldf_a
        self.compose_data_path = country_compose_data_path["california"]

    def _split_dataset(self, train_test_split_cmaq: dict, source_type: str):
        monitoring_whole_data = pd.read_csv(monitoring_data_path)[usa_whole_tags]
        source_cmaq, train_target_cmaq, valid_cmaq = train_test_split_cmaq['train_out_cluster'], train_test_split_cmaq['train_in_cluster'], train_test_split_cmaq['test_cluster']
        source_data = monitoring_whole_data.loc[np.all(np.isin(monitoring_whole_data[["cmaq_x", "cmaq_y"]], source_cmaq), axis=1)]
        train_target_data = monitoring_whole_data.loc[np.all(np.isin(monitoring_whole_data[["cmaq_x", "cmaq_y"]], train_target_cmaq), axis=1)]
        valid_data = monitoring_whole_data.loc[np.all(np.isin(monitoring_whole_data[["cmaq_x", "cmaq_y"]], valid_cmaq), axis=1)]
        source_data = source_select(source_data, source_type)
        return source_data, train_target_data, valid_data
    
    def _allocate_stations(self, source_dt: pd.DataFrame, train_target_dt: pd.DataFrame, valid_dt: pd.DataFrame):
        station_allocate = LdfInputCompose(source_dt, train_target_dt, valid_dt, self.station_num, "california", self.ldf_a)
        all_inputs, all_labels = station_allocate.allocate_all(False)
        return all_inputs, all_labels

    def save(self, number_of_splits: int, source_type: str):
        train_test_data_id = get_in_clusters(self.compose_data_path, self.target_station_num, number_of_splits)
        for split_id in train_test_data_id.keys():
            file_name = f"{self.compose_data_path}tl-cal-{self.target_station_num}/split-{split_id}/{source_type} nearest{self.station_num} dataset.npz"
            train_test_split_id = train_test_data_id[split_id]
            source_dt, train_target_dt, valid_dt = self._split_dataset(train_test_split_id, source_type)
            allocate_input, allocate_label = self._allocate_stations(source_dt, train_target_dt, valid_dt)
            np.savez(file_name,
                source_input = allocate_input["source"], 
                train_target_input = allocate_input["train_target"], 
                valid_input = allocate_input["valid"],
                source_label = allocate_label["source"], 
                train_target_label = allocate_label["train_target"], 
                valid_label = allocate_label["valid"])
            print(f"train_num:{self.target_station_num} source:{source_type} split{split_id} complete")

class LimaSplitLdfInputCompose:
    def __init__(self, target_station_num: int, station_num: int, ldf_a: bool):
        self.target_station_num = target_station_num
        self.station_num = station_num
        self.ldf_a = ldf_a

    def _split_dataset(self, train_test_split_coord: dict, source_type: str):
        monitoring_usa_data = pd.read_csv(monitoring_data_path)[transfer_tags["lima"]["source"]]
        monitoring_country_data = pd.read_csv(monitoring_country_data_path["lima"])[transfer_tags["lima"]["source"]]
        source_coords, train_target_coords, valid_coords = train_test_split_coord['train_out_cluster'], train_test_split_coord['train_in_cluster'], train_test_split_coord['test_cluster']
        source_data = monitoring_usa_data
        train_target_data = monitoring_country_data.loc[np.all(np.isin(monitoring_country_data[["lon", "lat"]], train_target_coords), axis=1)]
        valid_data = monitoring_country_data.loc[np.all(np.isin(monitoring_country_data[["lon", "lat"]], valid_coords), axis=1)]
        source_data = source_select(source_data, source_type)
        return source_data, train_target_data, valid_data
    
    def _allocate_stations(self, source_dt: pd.DataFrame, train_target_dt: pd.DataFrame, valid_dt: pd.DataFrame):
        station_allocate = LdfInputCompose(source_dt, train_target_dt, valid_dt, self.station_num, "lima", self.ldf_a)
        all_inputs, all_labels = station_allocate.allocate_all(False)
        return all_inputs, all_labels

    def save(self, number_of_splits: int, source_type: str):
        lima_data_path = country_compose_data_path["lima"]
        train_test_data_id = get_in_clusters(lima_data_path, self.target_station_num, number_of_splits)
        for split_id in train_test_data_id.keys():
            file_name = f"{lima_data_path}tl-cal-{self.target_station_num}/split-{split_id}/{source_type} nearest{self.station_num} dataset.npz"
            train_test_split_id = train_test_data_id[split_id]
            source_dt, train_target_dt, valid_dt = self._split_dataset(train_test_split_id, source_type)
            allocate_input, allocate_label = self._allocate_stations(source_dt, train_target_dt, valid_dt)
            np.savez(file_name,
                source_input = allocate_input["source"], 
                train_target_input = allocate_input["train_target"], 
                valid_input = allocate_input["valid"],
                source_label = allocate_label["source"], 
                train_target_label = allocate_label["train_target"], 
                valid_label = allocate_label["valid"])
            print(f"train_num:{self.target_station_num} source:{source_type} split{split_id} complete")
