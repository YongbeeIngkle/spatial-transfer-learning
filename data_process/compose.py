import numpy as np 
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from data_process.tag_info import earth_coord_tags, ldf_a_tags, transfer_ldf_input_tags

def get_in_clusters(data_path: str, train_num: int, set_num: int):
    """
    Get the train-target and test cmaq_id of target area for each split.

    data_path: path that cmaq information is saved
    train_num: number of train-target grids
    set_num: number of split
    """
    target_path = f"{data_path}tl-cal-{train_num}/"
    train_test_id = {}

    source_cmaq = np.load(f"{data_path}source_cmaq.npy")
    for set_id in range(set_num):
        target_cmaq = np.load(f"{target_path}split-{set_id}/target_cmaq.npz")
        train_cmaq = target_cmaq["train"]
        test_cmaq = target_cmaq["test"]
        train_test_id[set_id] = {
            "train_in_cluster":train_cmaq,
            "train_out_cluster":source_cmaq,
            "test_cluster":test_cmaq
        }
    return train_test_id

def source_select(source_original: pd.DataFrame, source_type: str):
    """
    Select the source based on the type.

    source_original: original source dataset
    source_type: type of source. ["east", "west", "east-north"].
    """
    if source_type == "east":
        source_dt = source_original[source_original["cmaq_x"] > 0]
    elif source_type == "east-north":
        source_dt = source_original[source_original["rid"] == 3]
    else:
        source_dt = source_original
    return source_dt

def _convert_loader(input_dt:np.ndarray, output_dt:np.ndarray, batch:int):
    """
    Convert input-output dataset as torch DataLoader.

    input_dt: input dataset
    output_dt: output dataset
    batch: batch size
    """
    if len(input_dt) < 1:
        raise Exception("input_dt length is 0.")
    if len(output_dt) < 1:
        raise Exception("output_dt length is 0.")
    dt_set = InputOutputSet(input_dt, output_dt)
    dt_loader = DataLoader(dt_set, batch_size=batch, shuffle=False, pin_memory=True)
    return dt_loader

def _set_pm25_statistic(mean: np.ndarray, std: np.ndarray, original_tags: list):
    mean_df = pd.DataFrame(mean, index=original_tags)
    std_df = pd.DataFrame(std, index=original_tags)
    if "pm25_value" in original_tags:
        mean_df.loc["pm25_value"] = 0
        std_df.loc["pm25_value"] = 1
    return np.array(mean_df), np.array(std_df)

def _select_ldf_tags(ldf_input: np.ndarray, original_tags: list, choose_tags: list):
    converted_data = []
    for station_id in range(ldf_input.shape[-1]):
        station_df = pd.DataFrame(ldf_input[:,:,station_id], columns=original_tags)
        converted_data.append(station_df[choose_tags])
    converted_data = np.stack(converted_data, -1)
    return converted_data

def _compose_ldf_a_set(ldf_input: np.ndarray, label_data: np.ndarray, original_tags: list):
    center_id = ldf_input.shape[-1]//2
    aod_values = _select_ldf_tags(ldf_input, original_tags, ['gc_aod'])
    center_aod_values = aod_values[:,:,center_id].flatten()
    new_label = np.vstack([center_aod_values, label_data]).T
    new_input = ldf_input.copy()
    aod_tag_id = np.arange(len(original_tags))[[f=="gc_aod" for f in original_tags]][0]
    new_input[:,aod_tag_id,center_id] = 0
    return new_input, new_label

class InputOutputSet(Dataset):
    def __init__(self, input_dt, output_dt):
        super().__init__()
        self.input_dt = torch.tensor(input_dt, dtype=torch.float32)
        self.output_dt = torch.tensor(output_dt, dtype=torch.float32)

    def __getitem__(self, i):
        return self.input_dt[i], self.output_dt[i]

    def __len__(self):
        return len(self.input_dt)

class LdfInputData:
    def __init__(self, transfer_name: str, source_type: str,  nearest_station_num: int, ldf_a: bool):
        self.source_type = source_type
        self.nearest_station_num = nearest_station_num
        self.ldf_a = ldf_a
        self.ldf_input_tags = transfer_ldf_input_tags[transfer_name]
        self.transfer_name = transfer_name
        self._define_shapes()

    def _define_shapes(self):
        if self.ldf_a:
            self.target_dim = len(ldf_a_tags)
        else:
            self.target_dim = 1
        self.input_shape = (len(self.ldf_input_tags), self.nearest_station_num+1)

    def _normalize_train_valid(self, source_input: np.ndarray, train_target_input: np.ndarray,  valid_input: np.ndarray):
        """
        Normalize the train (source and train target) and validation data.
        If standard deviation is 0, substitute to 1 to make the values 0 with just subtracting by mean.
        """
        train_input = np.vstack([source_input, train_target_input])
        mean, std = train_input.mean(axis=0), train_input.std(axis=0)
        mean, std = _set_pm25_statistic(mean, std, self.ldf_input_tags)
        std[std==0] = 1
        self.mean, self.std = mean, std
        source_dt = (source_input - mean) / std
        train_target_dt = (train_target_input - mean) / std
        valid_dt = (valid_input - mean) / std
        return source_dt, train_target_dt, valid_dt

    def read_data(self, file_name: str):
        save_npz = np.load(file_name)

        source_data, train_target_data, valid_data = save_npz["source_input"], save_npz["train_target_input"], save_npz["valid_input"]
        source_label, train_target_label, valid_label = save_npz["source_label"], save_npz["train_target_label"], save_npz["valid_label"]
        if self.ldf_a:
            source_data, source_label = _compose_ldf_a_set(source_data, source_label, self.ldf_input_tags)
            train_target_data, train_target_label = _compose_ldf_a_set(train_target_data, train_target_label, self.ldf_input_tags)
            valid_data, valid_label = _compose_ldf_a_set(valid_data, valid_label, self.ldf_input_tags)
        source_data, train_target_data, valid_data = self._normalize_train_valid(source_data, train_target_data, valid_data)
        source_set = {"input":source_data, "label":source_label}
        train_target_set = {"input":train_target_data, "label":train_target_label}
        valid_set = {"input":valid_data, "label":valid_label}
        return {"source":source_set, "train_target":train_target_set, "valid":valid_set}
    
    def convert_loader(self, train_unite: bool, file_name: str):
        """
        Convert the regression data to DataLoader. If unite the train dataset, unite the source and target train data as train data.

        train_unite: unite the train data or not
        """
        read_dataset = self.read_data(file_name)
        source_input = read_dataset["source"]["input"]
        source_label = read_dataset["source"]["label"]
        train_target_input = read_dataset["train_target"]["input"]
        train_target_label = read_dataset["train_target"]["label"]
        valid_input = read_dataset["valid"]["input"]
        valid_label = read_dataset["valid"]["label"]
        if train_unite:
            train_input = np.vstack([source_input, train_target_input])
            if len(source_label.shape) > 1:
                train_label = np.vstack([source_label, train_target_label])
            else:
                train_label = np.hstack([source_label, train_target_label])
            train_loader = _convert_loader(train_input, train_label, 128)
        else:
            source_loader = _convert_loader(source_input, source_label, 128)
            train_target_loader = _convert_loader(train_target_input, train_target_label, 128)
        valid_loader = _convert_loader(valid_input, valid_label, 128)
        if train_unite:
            return train_loader, valid_loader
        else:
            return source_loader, train_target_loader, valid_loader
        
    def compose_regress_input(self, file_name: str, all_features: dict):
        center_id = (self.nearest_station_num + 1) // 2
        read_dataset = self.read_data(file_name)
        source_input = read_dataset["source"]["input"][:,:,center_id]
        train_target_input = read_dataset["train_target"]["input"][:,:,center_id]
        valid_input = read_dataset["valid"]["input"][:,:,center_id]
        if all_features is not None:
            source_input[:,-1] = all_features["source"]
            train_target_input[:,-1] = all_features["train_target"]
            valid_input[:,-1] = all_features["valid"]
        source_data = {"input": source_input, "label": read_dataset["source"]["label"]}
        train_target_data = {"input": train_target_input, "label": read_dataset["train_target"]["label"]}
        valid_data = {"input": valid_input, "label": read_dataset["valid"]["label"]}
        return source_data, train_target_data, valid_data

    def read_pred_data(self, normalize: bool, train_source_file: str, pred_file: str):
        train_source_npz = np.load(train_source_file)
        pred_npz = np.load(pred_file)

        source_data, train_target_data, valid_data = train_source_npz["source_input"], train_source_npz["train_target_input"], pred_npz["valid_input"]
        source_label, train_target_label, valid_label = train_source_npz["source_label"], train_source_npz["train_target_label"], pred_npz["valid_label"]
        if normalize:
            source_data, train_target_data, valid_data = self._normalize_train_valid(source_data, train_target_data, valid_data)
        valid_set = {"input":valid_data, "label":valid_label}
        return valid_set
    
    def convert_pred_loader(self, train_source_path: str, pred_path: str):
        pred_dataset = self.read_pred_data(True, train_source_path, pred_path)
        valid_loader = _convert_loader(pred_dataset["input"], pred_dataset["label"], 128)
        return valid_loader
    
    def compose_pred_regress_input(self, train_source_path: str, pred_path: str, all_features: dict, input_normalize: bool):
        center_id = (self.nearest_station_num + 1) // 2
        pred_dataset = self.read_pred_data(input_normalize, train_source_path, pred_path)
        pred_input = pred_dataset["input"][:,:,center_id]
        if all_features is not None:
            pred_input[:,-1] = all_features
        if input_normalize:
            original_pred_dataset = self.read_pred_data(False, train_source_path, pred_path)
            original_input = original_pred_dataset["input"][:,:,center_id]
        else:
            original_input = pred_input
        coordinates = pd.DataFrame(original_input, columns=self.whole_ldf_tags)[earth_coord_tags[self.country_name]]
        pred_infoset = {"coord": coordinates, "input": pred_input}
        return pred_infoset
