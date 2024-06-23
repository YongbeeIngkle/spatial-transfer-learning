import os
import numpy as np
import pandas as pd
from data_process.compose import get_in_clusters
from data_process.tag_info import usa_whole_tags
from data_process.data_path import monitoring_country_data_path, country_daily_data_path, monitoring_country_data_path, country_compose_data_path
from data_process.pred_data import CaliforniaLdfInputCompose

def _whole_map_process():
    target_rid = 6
    usa_data_dir = country_daily_data_path["usa"]
    save_data_dir = country_daily_data_path["california"]
    all_files = sorted(os.listdir(usa_data_dir))
    for file in all_files:
        file_dt = pd.read_csv(usa_data_dir+file)
        file_dt = file_dt[usa_whole_tags]
        target_dt = file_dt[file_dt["rid"] == target_rid]
        target_dt.to_csv(f"{save_data_dir}{file}", index=False)

def _monitoring_process(target_station_num: int):
    target_rid = 6
    whole_monitoring_dt = pd.read_csv(monitoring_country_data_path["usa"])
    target_monitoring_dt = whole_monitoring_dt[whole_monitoring_dt['rid'] == target_rid][usa_whole_tags]
    target_monitoring_dt.to_csv(monitoring_country_data_path["california_whole"], index=False)
    target_number_cmaq_id = get_in_clusters(country_compose_data_path["california"], target_station_num, 20)
    train_monitoring_dt = target_monitoring_dt[np.isin(target_monitoring_dt["cmaq_id"], target_number_cmaq_id[0]["train_in_cluster"])]
    train_monitoring_dt.to_csv(monitoring_country_data_path["california9"], index=False)

if __name__ == "__main__":
    target_station_num = 9
    near_station_num = 12
    ldf_a = False
    source_type = "east-north"

    # _monitoring_process(target_station_num)
    # _whole_map_process()

    ldf_compose = CaliforniaLdfInputCompose(target_station_num, 0, near_station_num, ldf_a)
    ldf_compose.save(source_type)
