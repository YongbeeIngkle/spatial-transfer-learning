import os
import numpy as np
# import pandas as pd
# from data_process.tag_info import transfer_tags
# from data_process.data_path import country_daily_data_path, monitoring_country_data_path, country_compose_data_path
# from data_process.allocate import LdfInputCompose
# from model.regressor import CountryFeatureCompute

# class LimaLdfInputCompose:
#     def __init__(self, station_num: int, usa_source: str):
#         self.station_num = station_num
#         self.usa_source = usa_source

#     def _read_source_train(self):
#         usa_tags = transfer_tags["lima"]["source"]
#         usa_monitoring_data_path = monitoring_country_data_path["usa"]
#         lima_monitoring_data_path = monitoring_country_data_path["lima"]
#         monitoring_usa_data = pd.read_csv(usa_monitoring_data_path)[usa_tags]
#         if self.usa_source == "east":
#             monitoring_usa_data = monitoring_usa_data[monitoring_usa_data["cmaq_x"]>0]
#         monitoring_lima_data = pd.read_csv(lima_monitoring_data_path)[usa_tags]
#         return monitoring_usa_data, monitoring_lima_data
    
#     def _read_pred_data(self, file_name: str):
#         usa_tags = transfer_tags["lima"]["source"]
#         pred_country_data = pd.read_csv(country_daily_data_path["lima"]+file_name)[usa_tags]
#         return pred_country_data

#     def _allocate_stations(self, source_dt: pd.DataFrame, train_target_dt: pd.DataFrame, valid_dt: pd.DataFrame):
#         station_allocate = LdfInputCompose(source_dt, train_target_dt, valid_dt, self.station_num, "lima")
#         all_inputs, all_labels = station_allocate.allocate_all(False)
#         return all_inputs, all_labels

#     def save(self):
#         save_dir = f"{country_compose_data_path["lima"]}/"
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         source_dt, train_target_dt = self._read_source_train()
#         daily_files = os.listdir(country_daily_data_path["lima"])
#         daily_files.sort()
#         source_train_input, source_train_label = {"source": [], "train_target": []}, {"source": [], "train_target": []}
#         for date_file in daily_files:
#             original_file = date_file.split(".")[0]
#             valid_file_name = f"{save_dir}{original_file} nearest{self.station_num} dataset.npz"
#             pred_dt = self._read_pred_data(date_file)
#             allocate_input, allocate_label = self._allocate_stations(source_dt, train_target_dt, pred_dt)
#             if len(allocate_input) < 1:
#                 continue
#             np.savez(valid_file_name, valid_input = allocate_input["valid"], valid_label = allocate_label["valid"])
#             source_train_input["source"].append(allocate_input["source"])
#             source_train_input["train_target"].append(allocate_input["train_target"])
#             source_train_label["source"].append(allocate_label["source"])
#             source_train_label["train_target"].append(allocate_label["train_target"])
#             source_train_input["valid"] = allocate_input["valid"]
#             source_train_label["valid"] = allocate_label["valid"]
#             print(f"train_num:{date_file} complete")
#         source_train_file_name = f"{country_ldf_data_path["lima"]}/source-train nearest{self.station_num} dataset.npz"
#         np.savez(source_train_file_name, 
#                  source_input = np.vstack(source_train_input["source"]), 
#                  train_target_input = np.vstack(source_train_input["train_target"]), 
#                  source_label = np.hstack(source_train_label["source"]), 
#                  train_target_label = np.hstack(source_train_label["train_target"]),
#                  valid_input = source_train_input["valid"],
#                  valid_label=source_train_label["valid"])
        
#     def save_edit(self):
#         save_dir = f"{country_ldf_data_path["lima"]}/"
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         source_dt, train_target_dt = self._read_source_train()
#         daily_files = os.listdir(country_daily_data_path["lima"])
#         daily_files.sort()
#         source_train_input, source_train_label = {}, {}
#         for date_file in daily_files:
#             pred_dt = self._read_pred_data(date_file)
#             allocate_input, allocate_label = self._allocate_stations(source_dt, train_target_dt, pred_dt)
#             if len(allocate_input) < 1:
#                 continue
#             source_train_input["valid"] = allocate_input["valid"]
#             source_train_label["valid"] = allocate_label["valid"]
#             print(f"train_num:{date_file} complete")
#             break
#         source_train_file_name = f"{country_ldf_data_path["lima"]}/source-train nearest{self.station_num} dataset.npz"
#         load_dt = np.load(source_train_file_name)
#         np.savez(source_train_file_name, 
#                  source_input = load_dt["source_input"], 
#                  train_target_input = load_dt["train_target_input"], 
#                  source_label = load_dt["source_label"], 
#                  train_target_label = load_dt["train_target_label"],
#                  valid_input = source_train_input["valid"],
#                  valid_label=source_train_label["valid"])

class PredLdfSet:
    def __init__(self, country_name: str, source_type: str,  
                 nearest_station_num: int, feature_name: str, 
                 similarity_measure: str, coord_pm_input: bool, 
                 weighting_input: bool, contextual_ldf_target: bool,
                 input_normalize: bool):
        self.country_name = country_name
        self.source_type = source_type
        self.nearest_station_num = nearest_station_num
        self.feature_name = feature_name
        self.similarity_measure = similarity_measure
        self.coord_pm_input = coord_pm_input
        self.weighting_input = weighting_input
        self.contextual_ldf_target = contextual_ldf_target
        self.input_normalize = input_normalize
        self._list_ldf_files()
        self._define_feature_compute()
        self.file_id = 0

    def _list_ldf_files(self):
        self.country_file_path = f"{country_ldf_data_path[self.country_name]}{self.similarity_measure}/"
        file_name_character = f"nearest{self.nearest_station_num} dataset.npz"
        self.ldf_files = np.sort([x for x in os.listdir(self.country_file_path) if file_name_character in x])

    def _define_feature_compute(self):
        if self.feature_name == "NF":
            self.feature_compute = CountryFeatureCompute(
                self.country_name, self.source_type, self.nearest_station_num, self.feature_name, 
                self.similarity_measure, False, False, False
                )
        else:
            if self.feature_name == "LDF":
                self.feature_compute = CountryFeatureCompute(
                    self.country_name, self.source_type, self.nearest_station_num, self.feature_name, 
                    self.similarity_measure, self.coord_pm_input, self.weighting_input, 
                    self.contextual_ldf_target, False
                    )
            else:
                if self.feature_name == "DWA":
                    self.feature_compute = CountryFeatureCompute(
                        self.country_name, self.source_type, self.nearest_station_num, self.feature_name, "eu_distance", 
                        False, True, False
                        )
                elif self.feature_name == "SWA":
                    self.feature_compute = CountryFeatureCompute(
                        self.country_name, self.source_type, self.nearest_station_num, self.feature_name, "spatial_similarity", 
                        False, True, False
                        )

    def _process_data(self):
        pred_data_path = self.country_file_path + self.ldf_files[self.file_id]
        if self.feature_name == "NF":
            pred_feature = None
        elif self.feature_name == "LDF":
            pred_feature = self.feature_compute.compute_pred_ldf(pred_data_path)
        pred_set = self.feature_compute.combine_pred_input_feature(pred_data_path, pred_feature, self.input_normalize)
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
