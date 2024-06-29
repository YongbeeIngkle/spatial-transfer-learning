import os
import pandas as pd
from data_process.tag_info import transfer_tags
from data_process.data_path import monitoring_country_data_path
from data_process.pred_data import LimaLdfInputCompose

def _monitoring_process():
    whole_monitoring_dt = pd.read_csv("D:/Lima-data-info/lima_train.csv")
    lima_tags = transfer_tags["lima"]
    year16_dt = whole_monitoring_dt[whole_monitoring_dt['year'] == 2016][lima_tags["target"]]
    year16_dt.columns = lima_tags["source"]
    year16_dt.to_csv(monitoring_country_data_path["lima"], index=False)

if __name__ == "__main__":
    near_station_num = 12
    ldf_a = False
    source_type = "east-north"

    _monitoring_process()
    ldf_compose = LimaLdfInputCompose(near_station_num, ldf_a)
    ldf_compose.save_pred_daily(source_type)
    ldf_compose.save_monitor(source_type)
