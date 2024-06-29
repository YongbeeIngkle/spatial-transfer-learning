from data_process.pred_data import LimaLdfInputCompose

if __name__ == "__main__":
    near_station_num = 12
    ldf_a = False
    source_type = "whole"

    ldf_compose = LimaLdfInputCompose(near_station_num, ldf_a)
    ldf_compose.save_pred_daily(source_type)
    ldf_compose.save_monitor(source_type)
