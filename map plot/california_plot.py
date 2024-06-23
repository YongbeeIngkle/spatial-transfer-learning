import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_process.data_path import predict_result_path

def average_pm_values():
    # monitoring_dt = pd.read_csv("D:/US_PM25_data/california/california_station9_monitoring.csv")
    monitoring_dt = pd.read_csv("D:/US_PM25_data/california/california_whole_monitoring.csv")
    unique_coords = np.unique(monitoring_dt[["cmaq_x", "cmaq_y"]], axis=0)
    all_coord_pm = pd.DataFrame(unique_coords, columns=["cmaq_x", "cmaq_y"])
    all_coord_pm["pm25_value"] = np.zeros(len(all_coord_pm))
    for cmaq_x, cmaq_y in unique_coords:
        coord_data = monitoring_dt[(monitoring_dt["cmaq_x"]==cmaq_x) & (monitoring_dt["cmaq_y"]==cmaq_y)]
        all_coord_pm.loc[(all_coord_pm["cmaq_x"]==cmaq_x) & (all_coord_pm["cmaq_y"]==cmaq_y), "pm25_value"] = coord_data["pm25_value"].mean()
    return all_coord_pm

def _plot_california_map(pred_data_sample):
    coord_pm_mean = average_pm_values()
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.scatter(coord_pm_mean["cmaq_x"], coord_pm_mean["cmaq_y"], s=14, color="red", label="training PM 2.5 sensors")
    plt.scatter(pred_data_sample.index.get_level_values('cmaq_x'), pred_data_sample.index.get_level_values('cmaq_y'), color="gray", s=10, alpha=0.4, label="testing satellite locations")
    plt.legend(prop={'size': 11, 'weight':'bold'})
    plt.show()

def _interval_average(pred_files: list, start_date: int, end_date: int):
    file_num = 0
    all_preds = []
    for pred_date in range(start_date, end_date):
        date_pred_file = f"date{pred_date}.csv"
        if date_pred_file not in pred_files:
            continue
        date_pred_vals = pd.read_csv(f"{pred_dir}{date_pred_file}").set_index(["cmaq_x","cmaq_y"])
        date_pred_vals.columns = [f"pm25_date{pred_date}"]
        all_preds.append(date_pred_vals.copy())
        file_num += 1
    print(file_num)
    all_preds = pd.concat(all_preds, axis=1)
    pm_mean = all_preds.mean(axis=1)

    plt.figure(figsize=(6, 6))
    plt.scatter(pm_mean.index.get_level_values('cmaq_x'), pm_mean.index.get_level_values('cmaq_y'), c=pm_mean, s=10, cmap="rainbow", vmin=5, vmax=14, alpha=0.8)
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=17)
    plt.axis("off")
    # plt.show()
    plt.savefig(f'{pred_dir}figures/date{start_date}-{end_date} average.png')
    plt.close('all')

    # _plot_california_map(pm_mean)

if __name__=='__main__':
    model_name = "Nnw"
    feature_name = "LDF" ## what type of characeristic feature is to be produced -- SWA, DWA, LDF
    ldf_a = False
    target_station_num = 9
    nearest_station_num = 12
    split_id = 0
    result_path = predict_result_path["california"]

    pred_dir = f"{result_path}/{model_name}/{feature_name}/"
    pred_results = [f for f in os.listdir(pred_dir) if ".csv" in f]
    _interval_average(pred_results, 1, 366)
