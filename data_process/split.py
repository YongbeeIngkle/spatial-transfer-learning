import os
import pandas as pd
import numpy as np
from data_process.tag_info import usa_whole_tags, transfer_tags
from data_process.data_path import country_compose_data_path, monitoring_country_data_path
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
        target_cmaqs, target_count = np.unique(target_data["cmaq_id"], return_counts=True)
        source_cmaqs = np.unique(source_data["cmaq_id"])
        cmaq_set = {"source": source_cmaqs, "target": target_cmaqs}
        return cmaq_set, target_count
        
    def _read_input_label(self):
        """
        Read the input-label data and split into source and target cmaqs.
        """
        monitoring_whole_data = pd.read_csv(monitoring_country_data_path["usa"])[usa_whole_tags]
        source_target_set, target_count = self._source_target_split(monitoring_whole_data)
        return source_target_set, target_count
    
    def _split_data_coord(self, cmaq_ids: np.ndarray, train_num: int, split_num: int):
        """
        Split coordinate data into train-test dictionary.
        """
        np.random.seed(1000)
        set_dataset = {}
        for set_id in range(split_num):
            train_cmaqs = np.random.choice(cmaq_ids, train_num, replace=False)
            test_cmaqs = np.array([i for i in cmaq_ids if i not in train_cmaqs])
            set_dataset[set_id] = {"train": train_cmaqs, "test": test_cmaqs}
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

class CaliforniaSplitLdfInputCompose:
    def __init__(self, target_station_num: int, station_num: int, ldf_a: bool):
        self.target_station_num = target_station_num
        self.station_num = station_num
        self.ldf_a = ldf_a
        self.compose_data_path = country_compose_data_path["california"]

    def _split_dataset(self, train_test_split_id: dict, source_type: str):
        monitoring_whole_data = pd.read_csv(monitoring_country_data_path["usa"])[usa_whole_tags]
        source_index, train_target_index, valid_index = train_test_split_id['train_out_cluster'], train_test_split_id['train_in_cluster'], train_test_split_id['test_cluster']
        source_data = monitoring_whole_data.loc[np.isin(monitoring_whole_data["cmaq_id"], source_index)]
        train_target_data = monitoring_whole_data.loc[np.isin(monitoring_whole_data["cmaq_id"], train_target_index)]
        valid_data = monitoring_whole_data.loc[np.isin(monitoring_whole_data["cmaq_id"], valid_index)]
        source_data = source_select(source_data, source_type)
        return source_data, train_target_data, valid_data

    def _allocate_stations(self, source_dt: pd.DataFrame, train_target_dt: pd.DataFrame, valid_dt: pd.DataFrame):
        station_allocate = LdfInputCompose(source_dt, train_target_dt, valid_dt, self.station_num, "california", self.ldf_a)
        all_inputs, all_labels = station_allocate.allocate_all(True)
        return all_inputs, all_labels

    def save(self, number_of_splits: int, source_type: str):
        train_test_data_id = get_in_clusters(self.compose_data_path, self.target_station_num, number_of_splits)
        for split_id in train_test_data_id.keys():
            if self.ldf_a:
                file_name = f"{self.compose_data_path}tl-cal-{self.target_station_num}/split-{split_id}/{source_type} nearest{self.station_num} ldf_a dataset.npz"
            else:
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
