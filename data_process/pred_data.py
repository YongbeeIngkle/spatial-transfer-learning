import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from torch.utils.data import Dataset, DataLoader

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

def create_distance_matrix(dt1: pd.DataFrame, dt2: pd.DataFrame):
    """
    Create distance matrix between two dataframe. If the distance is 0 (i.e. same coordinate), replace to inf.

    dt1: first data
    dt2: second data
    """
    all_distance = distance_matrix(dt1, dt2)
    all_distance[all_distance==0] = np.inf
    all_distance = pd.DataFrame(all_distance, index=dt1.index, columns=dt2.index)
    return all_distance

class PredWeightAverage:
    def __init__(self, train_data:pd.DataFrame, train_label: pd.DataFrame, train_statistics: dict):
        """
        Compute weighted average.

        train_data: train input
        target_coord: target coordination
        trian_label: train label
        """
        self.train_data = train_data
        self.train_label = train_label
        self.train_statistics = train_statistics
        self.cmaq_cols = ["cmaq_x", "cmaq_y", "cmaq_id"]
        
    def allocate_weight(self, target_sample):
        """
        Allocate the weight (1/distance) for train data and validation data.
        """
        sample_normalize = (target_sample - self.train_statistics["mean"]) / self.train_statistics["std"]
        sample_normalize = sample_normalize.set_index("cmaq_id")
        train_cmaq = pd.DataFrame(np.unique(self.train_data[self.cmaq_cols], axis=0), columns=self.cmaq_cols).set_index("cmaq_id")
        self.target_weight = 1 / create_distance_matrix(sample_normalize[["cmaq_x", "cmaq_y"]], train_cmaq)

    def _date_weight_average(self, data: pd.DataFrame, weight: pd.DataFrame, train_label: pd.DataFrame, train_cmaq):
        """
        Compute weighted average of given data for one date.

        data: input data for computing weighted average
        weight: weight for weighted average
        train_label: label of train data
        train_cmaq: cmaq id of train data
        """
        exist_weight = weight.loc[data["cmaq_id"], np.isin(weight.columns, train_cmaq)]
        weight_label = train_label[exist_weight.columns]
        weight_sum = np.sum(exist_weight, axis=1)
        cmaq_wa = np.sum(exist_weight*weight_label, axis=1)/weight_sum
        cmaq_wa.index = data.index
        return cmaq_wa
        
    def compute_date_wa(self, target_data: pd.DataFrame):
        """
        Compute weighted average of train and validation data for given date.

        date: weighted average computing date
        """
        data_normalize = (target_data - self.train_statistics["mean"]) / self.train_statistics["std"]
        date = data_normalize["day"].iloc[0]
        date_train_data = self.train_data.loc[self.train_data["day"]==date].copy()
        date_train_label = self.train_label.loc[self.train_data["day"]==date]
        date_train_label.index = date_train_data["cmaq_id"]
        target_data_wa = self._date_weight_average(data_normalize, self.target_weight, date_train_label, date_train_data["cmaq_id"])
        return target_data_wa

class InputOutputSet(Dataset):
    def __init__(self, input_dt, output_dt):
        super().__init__()
        self.input_dt = input_dt
        self.output_dt = output_dt

    def __getitem__(self, i):
        return self.input_dt[i], self.output_dt[i]

    def __len__(self):
        return len(self.input_dt)

class EncodePredCompose:
    def __init__(self, source_type:str, statistics: dict, normalize=False, coord_pm=False):
        self.source_type = source_type
        self.statistics = statistics
        self.normalize = normalize
        self.coord_pm = coord_pm

    def read_encoder_data(self, date: int):
        target_data = np.load(f"D:/target-encode/tl-cal-15/split0/date{date}.npz")["dataset"]
        if self.coord_pm:
            target_data = target_data[:,[2,3,-1]]
        if self.normalize:
            target_data = (target_data - self.statistics["mean"]) / self.statistics["std"]
        return target_data

    def convert_loader(self, dataset: np.ndarray):
        data_loader = _convert_loader(dataset, np.zeros(len(dataset)), 64)
        return data_loader

class FeaturePredCompose:
    def __init__(self, source_type:str, statistics: dict, normalize=False, coord_pm=False):
        self.source_type = source_type
        self.statistics = statistics
        self.normalize = normalize
        self.coord_pm = coord_pm

    def combine_input_feature(self, input_data: pd.DataFrame, feature_data: np.ndarray):
        input_copy = input_data[self.statistics["mean"].index]
        if self.normalize:
            input_copy = (input_copy - self.statistics["mean"]) / self.statistics["std"]
        input_copy["feature"] = feature_data
        return input_copy
