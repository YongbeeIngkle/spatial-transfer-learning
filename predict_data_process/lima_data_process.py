import os
import pandas as pd
from data_process.tag_info import transfer_tags
from data_process.data_path import monitoring_country_data_path, country_daily_data_path
from data_process.pred_data import LimaLdfInputCompose

def _monitoring_process():
    whole_monitoring_dt = pd.read_csv("D:/Lima-data-info/lima_train.csv")
    lima_tags = transfer_tags["lima"]
    year16_dt = whole_monitoring_dt[whole_monitoring_dt['year'] == 2016][lima_tags["target"]]
    year16_dt.columns = lima_tags["source"]
    year16_dt.to_csv(monitoring_country_data_path["lima"], index=False)

def _whole_map_process():
    original_data_dir = "D:/Lima-data-info/Lima2016/"
    save_data_dir = country_daily_data_path["lima"]
    lima_tags = transfer_tags["lima"]
    all_files = os.listdir(original_data_dir)
    for file in all_files:
        file_dt = pd.read_csv(original_data_dir+file)
        file_dt = file_dt.rename(columns = {'PM25':'pm25_value'})
        file_dt = file_dt[lima_tags["target"]]
        file_dt.columns = lima_tags["source"]
        file_dt.to_csv(f"{save_data_dir}{file}", index=False)

if __name__ == "__main__":
    near_station_num = 12
    ldf_a = False
    source_type = "east-north"

    _monitoring_process()
    # _whole_map_process()

    ldf_compose = LimaLdfInputCompose(near_station_num, ldf_a)
    # ldf_compose.save_pred_daily(source_type)
    ldf_compose.save_monitor(source_type)
