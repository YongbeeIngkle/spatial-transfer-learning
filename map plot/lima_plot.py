import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from geopandas.tools import sjoin
from shapely.geometry import Point

def average_pm_values():
    monitoring_dt = pd.read_csv("D:/US_PM25_data/Lima/lima_monitoring.csv")
    unique_coords = np.unique(monitoring_dt[["lon", "lat"]], axis=0)
    all_coord_pm = pd.DataFrame(unique_coords, columns=["lon", "lat"])
    all_coord_pm["pm25_value"] = np.zeros(len(all_coord_pm))
    for lon, lat in unique_coords:
        coord_data = monitoring_dt[(monitoring_dt["lon"]==lon) & (monitoring_dt["lat"]==lat)]
        all_coord_pm.loc[(all_coord_pm["lon"]==lon) & (all_coord_pm["lat"]==lat), "pm25_value"] = coord_data["pm25_value"].mean()
    return all_coord_pm

def _plot_lima_map(inbound_lima_pm):
    coord_pm_mean = average_pm_values()
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.scatter(coord_pm_mean["lon"], coord_pm_mean["lat"], s=14, color="red", label="train sensors")
    plt.scatter(inbound_lima_pm.index.get_level_values('lon'), inbound_lima_pm.index.get_level_values('lat'), color="gray", s=10, alpha=0.4, label="satellite data")
    plt.legend(prop={'size': 11, 'weight':'bold'})
    plt.show()

def _find_inbound_lima(pm_mean: pd.DataFrame):
    peru_map_df = gpd.read_file('D:/US_PM25_data/Lima/Peru_shape2/PER_adm2.shp')
    lima_map_df = peru_map_df[peru_map_df['NAME_2'] == 'Lima']
    geometry = [Point(xy) for xy in zip(pm_mean.index.get_level_values('lon'), pm_mean.index.get_level_values('lat'))]
    gdf = gpd.GeoDataFrame(pm_mean.values, crs="EPSG:4326", geometry=geometry, columns=["pm25"])
    pointInPolys = sjoin(gdf, lima_map_df, how='left')
    grouped = pointInPolys.groupby('index_right')
    inbound_ids = np.hstack([list(grouped.groups[k]) for k in grouped.groups.keys()])
    return pm_mean.iloc[inbound_ids]

def _change_3digit(date_str: str):
    add0_num = 3 - len(date_str)
    convert_date = ''
    for i in range(add0_num):
        convert_date += '0'
    convert_date += date_str
    return convert_date

def _interval_average(pred_files: list, start_date: int, end_date: int):
    file_num = 0
    all_preds = []
    for pred_date in range(start_date, end_date):
        digit3_date = _change_3digit(str(pred_date))
        date_pred_file = [f for f in pred_files if f"_{digit3_date} nearest" in f]
        if len(date_pred_file) < 1:
            continue
        date_pred_vals = pd.read_csv(f"{pred_dir}{date_pred_file[0]}", index_col=0).set_index(["lon","lat"])
        date_pred_vals.columns = [f"pm25_date{pred_date}"]
        all_preds.append(date_pred_vals.copy())
        file_num += 1
    print(file_num)
    all_preds = pd.concat(all_preds, axis=1)
    pm_mean = all_preds.mean(axis=1)
    inbound_lima_pm = _find_inbound_lima(pm_mean)

    # plt.figure(figsize=(8, 8))
    # plt.scatter(pm_mean.index.get_level_values('lon'), pm_mean.index.get_level_values('lat'), c=pm_mean, s=6, cmap="rainbow", vmin=14.5, vmax=48.5)
    # plt.colorbar()
    # plt.axis("off")
    # plt.savefig(f'{pred_dir}date{start_date}-{end_date} average.png')
    # plt.close('all')

    plt.figure(figsize=(6, 6))
    plt.scatter(inbound_lima_pm.index.get_level_values('lon'), inbound_lima_pm.index.get_level_values('lat'), c=inbound_lima_pm, s=12, cmap="rainbow", vmin=14.5, vmax=48.5, marker="o", alpha=0.8)
    # cbar = plt.colorbar()
    # cbar.ax.tick_params(labelsize=17)
    plt.axis("off")
    # plt.show()
    plt.savefig(f'{pred_dir}date{start_date}-{end_date} average lima-bound.png')
    plt.close('all')

    # _plot_lima_map(inbound_lima_pm)

if __name__=='__main__':
    pred_dir = "D:/US_PM25_data/Lima/predictions/Nnw/LDF/n_dim_eu_distance/input_weight/contextual_input/"
    pred_results = [f for f in os.listdir(pred_dir) if ".csv" in f]
    _interval_average(pred_results, 0, 366)
