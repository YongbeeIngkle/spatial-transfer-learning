import os
import numpy as np
import pandas as pd
from data_process.distance import create_distance_matrix
from data_process.spatial_validation import get_clusters

def _drop_constant_col(train_dt: pd.DataFrame,):
    _std = train_dt.std(axis=0)
    train_dt_variable = train_dt.loc[:,_std>0]
    return train_dt_variable.columns

def _drop_na_col(train_dt: pd.DataFrame):
    train_drop_dt = train_dt.dropna(axis=1)
    return train_drop_dt.columns

def _drop_useless_col(source_data, train_target_data, valid_data):
    train_data = pd.concat([source_data, train_target_data])
    train_cols = _drop_na_col(train_data)
    train_cols = _drop_constant_col(train_data[train_cols])
    source_drop_const = source_data[train_cols]
    train_drop_const = train_target_data[train_cols]
    valid_drop_const = valid_data[train_cols]
    return source_drop_const, train_drop_const, valid_drop_const

def _sort_distance_stations(distance_data: pd.DataFrame):
    nearest_stations = pd.DataFrame(columns=range(distance_data.shape[1]), index=distance_data.index)
    for row in distance_data.index:
        nearest_stations.loc[row] = distance_data.columns[distance_data.loc[row].argsort()]
    return nearest_stations

class StationAllocate:
    def __init__(self, source_data, train_target_data, valid_data, source_label, train_target_label, station_num, save_path):
        self.source_data = source_data
        self.train_target_data = train_target_data
        self.valid_data = valid_data
        self.source_label = source_label
        self.train_target_label = train_target_label
        self.station_num = station_num
        self.save_path = save_path
        self.cmaq_cols = ["cmaq_x", "cmaq_y", "cmaq_id"]
        self._compute_distances()
        self.source_sort_stations = _sort_distance_stations(self.source_distance)
        self.train_target_sort_stations = _sort_distance_stations(self.train_target_distance)
        self.valid_sort_stations = _sort_distance_stations(self.valid_distance)
        self._allocate_all_data()

    def _compute_distances(self):
        source_cmaq = pd.DataFrame(np.unique(self.source_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        train_target_cmaq = pd.DataFrame(np.unique(self.train_target_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        train_cmaq = pd.concat([source_cmaq, train_target_cmaq])
        valid_cmaq = pd.DataFrame(np.unique(self.valid_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        self.source_distance = create_distance_matrix(source_cmaq, train_cmaq)
        self.train_target_distance = create_distance_matrix(train_target_cmaq, train_cmaq)
        self.valid_distance = create_distance_matrix(valid_cmaq, train_cmaq)

    def _date_allocate_data(self, data: pd.DataFrame, train_data: pd.DataFrame, sort_stations: pd.DataFrame, train_label):
        cmaq_id_data = data.set_index("cmaq_id")
        cmaq_id_data["PM25"] = 0
        cmaq_id_train = train_data.set_index("cmaq_id")
        cmaq_id_train["PM25"] = train_label
        date_stations = sort_stations.loc[cmaq_id_data.index]

        date_exist_stations = []
        for id in date_stations.index:
            row_station = date_stations.loc[id]
            row_exist_stations = row_station[np.isin(row_station, cmaq_id_train.index)].reset_index(drop=True)
            date_exist_stations.append(row_exist_stations)
        date_exist_stations = pd.concat(date_exist_stations, axis=1).T

        station_data = []
        for s in range(self.station_num):
            near_data = cmaq_id_train.loc[date_exist_stations[s]]
            station_data.append(near_data)
        station_data.insert(len(station_data)//2, cmaq_id_data)
        stack_station_data = np.stack(station_data, -1)
        return stack_station_data
        
    def _compute_date_set(self, date):
        date_source_data = self.source_data.loc[self.source_data["day"]==date].copy()
        date_train_target_data = self.train_target_data.loc[self.train_target_data["day"]==date].copy()
        date_valid_data = self.valid_data.loc[self.valid_data["day"]==date].copy()
        date_source_label = self.source_label.loc[date_source_data.index]
        date_train_target_label = self.train_target_label.loc[date_train_target_data.index]
        date_train_data = pd.concat([date_source_data, date_train_target_data])
        date_train_label = pd.concat([date_source_label, date_train_target_label])
        date_train_label.index = date_train_data["cmaq_id"]
        date_valid_dataset = self._date_allocate_data(date_valid_data, date_train_data, self.valid_sort_stations, date_train_label)
        np.savez(f"{self.save_path}date{date}.npz", dataset=date_valid_dataset)

    def _allocate_all_data(self):
        all_dates = np.unique(self.valid_data["day"])
        for date in all_dates:
            print(f"date {date}")
            self._compute_date_set(date)

def _get_target_cmaqs(monitoring_whole_data, coord_whole_data: pd.DataFrame, target_cluster: int):
    whole_cmaq = coord_whole_data.drop_duplicates().reset_index(drop=True)[['cmaq_x', 'cmaq_y', "cmaq_id"]]
    input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
    label_dt = monitoring_whole_data["pm25_value"]
    _, cluster_model = get_clusters(input_dt, label_dt)
    whole_clsuter = cluster_model.predict(whole_cmaq[['cmaq_x', 'cmaq_y']])
    target_cmaq = whole_cmaq.iloc[whole_clsuter==target_cluster]
    return target_cmaq["cmaq_id"]

class InputComposer:
    def __init__(self, train_num):
        self.train_num = train_num

        self._set_paths()

    def _set_paths(self):
        self.whole_monitoring_path = "data/us_monitoring.csv"
        self.whole_coord_path = "data/largeUS_coords_pred.csv"
        self.daily_path = "D:/split-by-day/"

    def target_data_compose(self, area_id: int):
        monitoring_whole_data = pd.read_csv(self.whole_monitoring_path)[tag_names]
        coord_whole_data = pd.read_csv(self.whole_coord_path, index_col=0)
        target_cmaqs = _get_target_cmaqs(monitoring_whole_data, coord_whole_data, area_id)

        for date in range(1, 366):
            save_path = f"{self.daily_path}us-2011-satellite-target{area_id}-day-{date}.csv"
            if os.path.exists(save_path):
                continue
            print(f"date {date} target dataset compose")
            whole_date_data = pd.read_csv(f"{self.daily_path}us-2011-satellite-day-{date}.csv")[tag_names[:-1]]
            target_date_data = whole_date_data[np.isin(whole_date_data["cmaq_id"], target_cmaqs)]
            target_date_data.to_csv(save_path, index=False)

class LdfInputComposer:
    def __init__(self, area_id: int, split_id: int, train_num: int):
        self.area_id = area_id
        self.split_id = split_id
        self.train_num = train_num
        self.cmaq_cols = ["cmaq_x", "cmaq_y", "cmaq_id"]

        self._set_paths()
        self._define_data()
        self._split_train_valid_cmaq()

    def _set_paths(self):
        self.monitor_data_path = "data/split-data/"
        self.daily_path = "D:/split-by-day/"
        self.whole_coord_path = "data/largeUS_coords_pred.csv"
        self.compose_path = f"D:/target-encode/tl-cal-{self.train_num}/split{self.split_id}/"

    def _define_data(self):
        self.train_test_data_id = get_in_clusters(self.monitor_data_path, self.train_num)[self.split_id]
        monitoring_whole_data = pd.read_csv("data/us_monitoring.csv")[tag_names]
        self.input_dt = monitoring_whole_data.drop(columns=["pm25_value"])
        self.label_dt = monitoring_whole_data["pm25_value"]
        target_data = []
        for date in range(1,366):
            target_date_data = pd.read_csv(f"D:/split-by-day/us-2011-satellite-target{self.area_id}-day-{date}.csv")
            target_data.append(target_date_data)
        self.target_data = pd.concat(target_data)

    def _split_train_valid_cmaq(self):
        source_index, train_target_index = self.train_test_data_id['train_out_cluster'], self.train_test_data_id['train_in_cluster']
        source_input, source_label = self.input_dt.loc[np.isin(self.input_dt["cmaq_id"], source_index)], self.label_dt[np.isin(self.input_dt["cmaq_id"], source_index)]
        train_target_input, train_target_label = self.input_dt.loc[np.isin(self.input_dt["cmaq_id"], train_target_index)], self.label_dt[np.isin(self.input_dt["cmaq_id"], train_target_index)]
        source_input, train_target_input, self.target_data = _drop_useless_col(source_input, train_target_input, self.target_data)
        self.source_dt = {"input":source_input, "label": source_label}
        self.train_target_dt = {"input":train_target_input, "label": train_target_label}

    def allocate_near_data(self):
        source_input = self.source_dt["input"]
        train_target_input = self.train_target_dt["input"]
        source_label = self.source_dt["label"]
        train_target_label = self.train_target_dt["label"]
        StationAllocate(source_input, train_target_input, self.target_data, source_label, train_target_label, 12, self.compose_path)

if __name__=='__main__':
    train_number = 15
    area_id = 4
    split_id = 0

    composer = InputComposer(train_number)
    composer.target_data_compose(area_id)

    ldf_input_composer = LdfInputComposer(area_id, split_id, train_number)
    ldf_input_composer.allocate_near_data()
