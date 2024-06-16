import numpy as np
import pandas as pd
from data_process.tag_info import transfer_spatial_coord_tags, transfer_ldf_input_tags, ldf_a_tags
from data_process.distance import create_distance_matrix, create_distance_pairwise

def _normalize_data(dataset: pd.DataFrame, mean: pd.Series, std: pd.Series):
    return (dataset-mean) / std

def _sort_distance_stations(distance_data: pd.DataFrame):
    """
    Sort the columns of distance matrix row by row.

    distance_data: distance matrix data
    """
    nearest_stations = pd.DataFrame(columns=range(distance_data.shape[1]), index=distance_data.index)
    for row in distance_data.index:
        nearest_stations.loc[row] = distance_data.columns[distance_data.loc[row].argsort()]
    return nearest_stations

def _select_ldf_tags(ldf_input: np.ndarray, original_tags: list, choose_tags: list):
    converted_data = []
    for station_id in range(ldf_input.shape[-1]):
        station_df = pd.DataFrame(ldf_input[:,:,station_id], columns=original_tags)
        converted_data.append(station_df[choose_tags])
    converted_data = np.stack(converted_data, -1)
    return converted_data

def _spatially_weight(ldf_input: np.ndarray, original_tags: list, compute_tags: list):
    """
    Weight the LDF input with xy-coordinate distance.
    """
    center_id = ldf_input.shape[-1]//2
    computing_tags_ldf = _select_ldf_tags(ldf_input, original_tags, compute_tags)
    center_tags_ldf = np.expand_dims(computing_tags_ldf[:,:,center_id], -1)
    distances = create_distance_pairwise(computing_tags_ldf, center_tags_ldf)

    # Compute weight sum excluding center(=0)
    center_off_weights = 1/np.delete(distances, center_id, 1)
    sum_weights = center_off_weights.sum(axis=1)

    weighted_input = ldf_input.copy()
    # Weight the input values of each station, except the center station data.
    for p in range(ldf_input.shape[2]):
        if p == center_id:
            continue
        distance_weight = (1/distances[:,p])/sum_weights
        distance_weight = np.expand_dims(distance_weight, axis=-1)
        weighted_input[:,:,p] = ldf_input[:,:,p]*distance_weight
    return weighted_input

class LdfInputCompose:
    def __init__(self, source_data: pd.DataFrame, train_target_data: pd.DataFrame, valid_data: pd.DataFrame,
                 station_num: int, country_name: str, ldf_a: bool):
        """
        Allocate nearest monitoring station dataset.

        source_data: source input data
        train_target_data: train target input data
        valid_data: validation input data
        station_num: number of allocating nearest monitoring data
        """
        self.source_data = source_data
        self.train_target_data = train_target_data
        self.valid_data = valid_data
        self.station_num = station_num
        self.ldf_input_tags = transfer_ldf_input_tags[country_name][:]
        self.measure_tags = transfer_spatial_coord_tags[country_name]
        self.ldf_a = ldf_a
        if ldf_a:
            self.ldf_input_tags.remove("gc_aod")
        self._compute_train_statistics()

    def _compute_train_statistics(self):
        train_data = pd.concat([self.source_data, self.train_target_data])
        self.train_mean, self.train_std = train_data.mean(), train_data.std()

    def _extract_date_data(self, date: str, same_territory: bool):
        if same_territory:
            date_source_data = self.source_data[self.source_data["day"]==date].set_index("cmaq_id", drop=True)
            date_train_target_data = self.train_target_data[self.train_target_data["day"]==date].set_index("cmaq_id", drop=True)
            date_valid_data = self.valid_data.loc[self.valid_data["day"]==date].set_index("cmaq_id", drop=True)
        else:
            date_valid_data = self.valid_data.loc[self.valid_data["day"]==date]
            date_train_target_data = self.train_target_data.loc[self.train_target_data["day"]==date]
            date_source_data = self.source_data.loc[self.source_data["day"]==date]
            date_source_data.index = range(len(date_source_data))
            date_train_target_data.index = range(len(date_source_data), len(date_source_data)+len(date_train_target_data))
            date_valid_data.index = range(len(date_source_data)+len(date_train_target_data), len(date_source_data)+len(date_train_target_data)+len(date_valid_data))
        return date_source_data, date_train_target_data, date_valid_data

    def _get_nearest_stations(self, compute_input: pd.DataFrame, train_input: pd.DataFrame):
        mean, std = self.train_mean[self.measure_tags], self.train_std[self.measure_tags]
        compute_input_scaled = _normalize_data(compute_input[self.measure_tags], mean, std)
        train_input_scaled = _normalize_data(train_input[self.measure_tags], mean, std)
        distance = create_distance_matrix(compute_input_scaled, train_input_scaled)
        nearest_stations = _sort_distance_stations(distance)
        return nearest_stations
    
    def _data_compose_ldf(self, compute_input: pd.DataFrame, train_input: pd.DataFrame):
        nearest_stations = self._get_nearest_stations(compute_input, train_input)
        
        date_exist_stations = []
        for id in nearest_stations.index:
            row_station = nearest_stations.loc[id]
            row_exist_stations = row_station[np.isin(row_station, train_input.index)].reset_index(drop=True)
            date_exist_stations.append(row_exist_stations)
        date_exist_stations = pd.concat(date_exist_stations, axis=1).T

        ### Make a set (station_data) which has 12 optimal neighbors
        station_data = []
        for s in range(self.station_num):
            near_data = train_input.loc[date_exist_stations[s]]
            station_data.append(near_data)

        ### Insert the main station in the middle to create a stack with 6 stations above and 6 below
        compute_input_edit = compute_input.copy()
        compute_input_edit["pm25_value"] = 0
        station_data.insert(len(station_data)//2, compute_input_edit)
        stack_station_data = np.stack(station_data, -1)
        stack_station_data = _spatially_weight(stack_station_data, self.ldf_input_tags, self.measure_tags)

        ### Check Sanity of composed dataset.
        if stack_station_data.shape[1] != len(self.ldf_input_tags):
            raise Exception("The tags should be same as LDF input tags.")
        return stack_station_data
    
    def _compose_ldf_input(self, source_input: pd.DataFrame, train_target_input: pd.DataFrame, valid_input: pd.DataFrame):
        train_input = pd.concat([source_input, train_target_input])
        source_ldf_input = self._data_compose_ldf(source_input, train_input)
        if len(valid_input) > 0:
            valid_ldf_input = self._data_compose_ldf(valid_input, train_input)
        else:
            valid_ldf_input = []
        if len(train_target_input) > 0:
            train_target_ldf_input = self._data_compose_ldf(train_target_input, train_input)
        else:
            train_target_ldf_input = []
        return source_ldf_input, train_target_ldf_input, valid_ldf_input

    def allocate_all(self, same_territory: bool):
        all_dates = np.unique(self.valid_data["day"])
        all_source_data, all_train_target_data, all_valid_data = [], [], []
        all_source_label, all_train_target_label, all_valid_label = [], [], []
        for date_num, date in enumerate(all_dates):
            print(f"date {date_num}")
            date_source, date_train_target, date_valid = self._extract_date_data(date, same_territory)
            if len(date_source) < 1:
                continue
            if self.ldf_a:
                source_label, train_target_label, valid_label = date_source[ldf_a_tags], date_train_target[ldf_a_tags], date_valid[ldf_a_tags]
                date_source = date_source.drop(columns=['gc_aod'])
                date_train_target = date_train_target.drop(columns=['gc_aod'])
                date_valid = date_valid.drop(columns=['gc_aod'])
            else:
                source_label, train_target_label, valid_label = date_source[["pm25_value"]], date_train_target[["pm25_value"]], date_valid[["pm25_value"]]
            source_input, train_target_input, valid_input = self._compose_ldf_input(date_source, date_train_target, date_valid)
            all_source_data.append(source_input)
            all_source_label.append(source_label)
            if len(train_target_input) > 0:
                all_train_target_data.append(train_target_input)
                all_train_target_label.append(train_target_label)
            all_valid_data.append(valid_input)
            all_valid_label.append(valid_label)
        if len(all_train_target_data) < 1:
            return [], []
        all_source_data = np.vstack(all_source_data)
        all_train_target_data = np.vstack(all_train_target_data)
        all_valid_data = np.vstack(all_valid_data)
        all_source_label = np.vstack(all_source_label)
        all_train_target_label = np.vstack(all_train_target_label)
        all_valid_label = np.vstack(all_valid_label)
        composed_inputs = {
            "source":all_source_data, 
            "train_target":all_train_target_data, 
            "valid":all_valid_data
            }
        composed_labels = {
            "source":all_source_label, 
            "train_target":all_train_target_label, 
            "valid":all_valid_label
            }
        return composed_inputs, composed_labels
