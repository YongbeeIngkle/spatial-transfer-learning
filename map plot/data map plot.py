import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_process.data_path import pred_whole_us_path, monitoring_data_path, compose_data_path
from data_process.tag_info import whole_tag_names
from data_process.compose import get_in_clusters

def _source_target_split(coord_cluster, rid):
    target_coords = coord_cluster.loc[coord_cluster["rid"] == rid]
    source_coords = coord_cluster.loc[coord_cluster["rid"] != rid]
    return {"source": source_coords, "target": target_coords}

def _save_whole_coord_rid():
    day1_data = pd.read_csv("D:/California-Encoder-Transfer/data/satellite/us-2011-satellite-day-1.csv")
    coord_whole_data = day1_data[["cmaq_x", "cmaq_y", "cmaq_id", "rid"]]
    coord_whole_data.to_csv(pred_whole_us_path, index=False)

class SingleAnalyzerClimate:
    def __init__(self, monitoring_data, whole_coord_rid, train_num):
        self.monitoring_data = monitoring_data
        self.whole_coord_rid = whole_coord_rid
        self.train_num = train_num
        self.target_rid = 6
        self.east_north_rid = 3

    def get_target_coords(self):
        train_test_data_id = get_in_clusters(compose_data_path, self.train_num, 20)
        monitoring_whole_data = pd.read_csv(monitoring_data_path)[whole_tag_names]
        all_target_sets = {}
        for split_id in train_test_data_id.keys():
            train_test_split_id = train_test_data_id[split_id]
            source_index, train_target_index, valid_index = train_test_split_id['train_out_cluster'], train_test_split_id['train_in_cluster'], train_test_split_id['test_cluster']
            train_target_data = monitoring_whole_data.loc[np.isin(monitoring_whole_data["cmaq_id"], train_target_index)]
            valid_data = monitoring_whole_data.loc[np.isin(monitoring_whole_data["cmaq_id"], valid_index)]
            all_target_sets[split_id] = {"train": train_target_data, "test": valid_data}
        return all_target_sets

    def _split_data_coord(self):
        source_target_set = _source_target_split(self.monitoring_data, self.target_rid)
        train_test_coord = self.get_target_coords()
        return source_target_set["source"], train_test_coord

    def plot_whole_rid(self, save, alpha):
        plt.figure(figsize=(13,8))
        for rid in np.sort(np.unique(self.whole_coord_rid["rid"])):
            cluster_coords = self.whole_coord_rid[self.whole_coord_rid["rid"]==rid]
            if rid == self.target_rid:
                plt.scatter(cluster_coords['cmaq_x'], cluster_coords['cmaq_y'], s=3, alpha=alpha, color="brown")
            elif rid == self.east_north_rid:
                plt.scatter(cluster_coords['cmaq_x'], cluster_coords['cmaq_y'], s=3, alpha=alpha, color="orange")
            else:
                east_coords = cluster_coords[cluster_coords['cmaq_x'] > 0]
                west_coords = cluster_coords[cluster_coords['cmaq_x'] < 0]
                plt.scatter(east_coords['cmaq_x'], east_coords['cmaq_y'], s=3, alpha=alpha, color="olive")
                plt.scatter(west_coords['cmaq_x'], west_coords['cmaq_y'], s=3, alpha=alpha, color="gray")
        if save:
            plt.show()
    
    def plot_target(self, source_type="whole"):
        marker_size = 16
        save_dir = f"figures/target cluster split/tl-cal-{self.train_num}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        source_coord, target_coord_set = self._split_data_coord()
        if source_type == "east half":
            save_dir = f"{save_dir}east half "
            east_coord = source_coord[(source_coord["cmaq_x"] > 0) & (source_coord["rid"] != self.east_north_rid)]
            west_coord = source_coord[source_coord["cmaq_x"] < 0]
            east_north_coord = source_coord[source_coord["rid"] == self.east_north_rid]
        if source_type == "east-north":
            save_dir = f"{save_dir}east-north "
            east_coord = source_coord[source_coord["rid"] == self.east_north_rid]
        for set_id in target_coord_set.keys():
            set_data = target_coord_set[set_id]
            train_coord = set_data["train"]
            test_coord = set_data["test"]
            fig_savedir = f"{save_dir}split-{set_id}.png"
            self.plot_whole_rid(False, 0.1)
            plt.scatter(train_coord["cmaq_x"], train_coord["cmaq_y"], s=marker_size, color="green", alpha=0.6, marker='o', label="train-target")
            plt.scatter(test_coord["cmaq_x"], test_coord["cmaq_y"], s=marker_size, color='red', alpha=0.6, marker='^', label="test")
            if "east" in source_type:
                plt.scatter(west_coord["cmaq_x"], west_coord["cmaq_y"], s=marker_size, color='blue', alpha=1, marker="v", label="source")
                plt.scatter(east_coord["cmaq_x"], east_coord["cmaq_y"], s=marker_size, color='purple', alpha=1, marker="x", label="east source")
                plt.scatter(east_north_coord["cmaq_x"], east_north_coord["cmaq_y"], s=marker_size, color='royalblue', alpha=1, marker=">", label="east-north source")
            if source_type == "east half":
                plt.axvline(0, linestyle='-', color='r', alpha=0.4)
            plt.legend(bbox_to_anchor=(0.65, 1.05), loc="upper left", prop={'size': 11, 'weight':'bold'})
            plt.axis('off')
            # plt.show()
            plt.savefig(fig_savedir)
            plt.close()

if __name__=='__main__':
    sp = 15
    # _save_whole_coord_rid()
    monitoring_whole_data = pd.read_csv(monitoring_data_path)[whole_tag_names]
    coord_whole_data = pd.read_csv(pred_whole_us_path)
    whole_coord_rid = coord_whole_data.drop_duplicates().reset_index(drop=True)[['cmaq_x', 'cmaq_y', 'rid']]

    single_analyzer = SingleAnalyzerClimate(monitoring_whole_data, whole_coord_rid, sp)
    # single_analyzer.plot_whole_rid(True, 0.1)
    single_analyzer.plot_target("east half")
