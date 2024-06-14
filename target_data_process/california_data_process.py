import os
import numpy as np
import pandas as pd
from data_process.compose import get_in_clusters
from data_process.tag_info import usa_whole_tags
from data_process.data_path import monitoring_data_path, country_daily_data_path, daily_whole_data_path, monitoring_country_data_path, country_compose_data_path

def _whole_map_process():
    target_rid = 6
    save_data_dir = country_daily_data_path["california"]
    all_files = os.listdir(daily_whole_data_path)
    for file in all_files:
        file_dt = pd.read_csv(daily_whole_data_path+file)
        file_dt = file_dt[usa_whole_tags]
        target_dt = file_dt[file_dt["rid"] == target_rid]
        target_dt.to_csv(f"{save_data_dir}{file}", index=False)

def _monitoring_process():
    target_rid = 6
    whole_monitoring_dt = pd.read_csv(monitoring_data_path)
    target_monitoring_dt = whole_monitoring_dt[whole_monitoring_dt['rid'] == target_rid][usa_whole_tags]
    target_monitoring_dt.to_csv("D:/US_PM25_data/california/california_whole_monitoring.csv", index=False)
    target_number_cmaq_id = get_in_clusters(country_compose_data_path["california"], 9, 20)
    train_monitoring_dt = target_monitoring_dt[np.isin(target_monitoring_dt["cmaq_id"], target_number_cmaq_id[0]["train_in_cluster"])]
    train_monitoring_dt.to_csv(monitoring_country_data_path["california"], index=False)

if __name__ == "__main__":
    _monitoring_process()
    _whole_map_process()
