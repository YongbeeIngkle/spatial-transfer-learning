import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

def _split_in_cluster(coordinates: pd.DataFrame):
    """
    Split train-test coordinates in a cluster.

    coordinates: xy-coordinate dataset
    """
    if "cmaq_x" not in coordinates.columns:
        raise Exception("'cmaq_x' must be in the input coordinate columns.")
    if "cmaq_y" not in coordinates.columns:
        raise Exception("'cmaq_y' must be in the input coordinate columns.")

    unique_coors, coor_nums = np.unique(coordinates, axis=0, return_counts=True)
    unique_coors = pd.DataFrame(unique_coors, columns=coordinates.columns)
    large_num_coors = unique_coors[coor_nums>=np.percentile(coor_nums,60)]
    train_idx = np.random.choice(large_num_coors.index, size=len(coor_nums)//10)
    train_coors =  unique_coors.loc[train_idx]
    test_coors =  unique_coors.drop(train_idx)
    return train_coors, test_coors

def _split_train_test(xy_cluster: pd.DataFrame):
    """
    Split train-test coordinates in all clusters.

    xy_cluster: xy-coordinate and cluster data
    """
    if type(xy_cluster) is not pd.DataFrame:
        raise Exception("xy_cluster data is not pd.DataFrame.")

    np.random.seed(1000)
    unique_xy_cluster = xy_cluster.drop_duplicates()
    all_split_grids, all_split_data_id = {}, {}
    for cluster in np.sort(pd.unique(unique_xy_cluster["cluster id"])):
        cluster_xy = xy_cluster.loc[xy_cluster["cluster id"]==cluster, ["cmaq_x","cmaq_y"]]
        out_cluster_xy = unique_xy_cluster.loc[unique_xy_cluster["cluster id"]!=cluster, ["cmaq_x","cmaq_y"]].drop_duplicates()
        cluster_train, cluster_test = _split_in_cluster(cluster_xy)
        all_split_grids[cluster] = {
            "train_in_cluster":cluster_train,
            "train_out_cluster":out_cluster_xy,
            "test_cluster":cluster_test
        }
        all_split_data_id[cluster] = {
            "train_in_cluster":xy_cluster.index[np.isin(xy_cluster[["cmaq_x","cmaq_y"]], cluster_train).min(axis=1)],
            "train_out_cluster":xy_cluster.index[np.isin(xy_cluster[["cmaq_x","cmaq_y"]], out_cluster_xy).min(axis=1)],
            "test_cluster":xy_cluster.index[np.isin(xy_cluster[["cmaq_x","cmaq_y"]], cluster_test).min(axis=1)]
        }
    return all_split_grids, all_split_data_id

def get_clusters(input_dt: pd.DataFrame, label_dt: pd.Series):
    """
    Get the cluster grids of USA area.
    """
    single_grid = ClusterGrid("KMeans")
    whole_cluster, _ = single_grid.cluster_grids(input_dt, pd.Series(label_dt))
    _, train_test_data_id = single_grid.split_train_test(input_dt, whole_cluster)
    return train_test_data_id, single_grid.cluster_model

class ClusterGrid:
    def __init__(self, cluster_method="GaussianMixture", cluster_num=10):
        """
        Cluster Model for whole USA PM2.5 monitoring station grids (xy coordinates).

        cluster_method: clustering method. One of ["GaussianMixture", "KMeans"]
        cluster_num: number of cluster
        """
        self.cluster_num = cluster_num
        self.cluster_method = cluster_method

    def _train_cluster_model(self, coordinates: pd.DataFrame, method: str, n_clusters: int):
        """
        Train the clustering model.

        coordinates: xy-coordinates of monitoring stations
        method: clustering method
        n_clusters: number of clusters
        """
        np.random.seed(1000)
        if method == 'GaussianMixture':
            self.cluster_model = GaussianMixture(n_components=n_clusters).fit(coordinates)
        elif method == "KMeans":
            self.cluster_model = KMeans(n_clusters=n_clusters, n_init="auto").fit(coordinates)

    def _cluster_coords(self, coordinates: pd.DataFrame, method: str, n_clusters: int):
        """
        Train clustering model and compute the cluster results of whole and unique coordinates.
        """
        if "cmaq_x" not in coordinates.columns:
            raise Exception("'cmaq_x' must be in the input coordinate columns.")
        if "cmaq_y" not in coordinates.columns:
            raise Exception("'cmaq_y' must be in the input coordinate columns.")
        if "cmaq_id" not in coordinates.columns:
            raise Exception("'cmaq_id' must be in the input coordinate columns.")
        if method not in ["GaussianMixture", "KMeans"]:
            raise Exception("Inappropriate method.")

        unique_coors = coordinates.drop_duplicates()
        self._train_cluster_model(unique_coors[["cmaq_x", "cmaq_y"]], method, n_clusters)
        coor_pred = self.cluster_model.predict(unique_coors[["cmaq_x", "cmaq_y"]])
        whole_pred = self.cluster_model.predict(coordinates[["cmaq_x", "cmaq_y"]])
        coor_clusters = unique_coors.copy()
        coor_clusters["cluster id"] = coor_pred
        whole_df = pd.DataFrame(whole_pred, index=coordinates.index, columns=["cluster id"])
        return whole_df, coor_clusters

    def cluster_grids(self, input_dt: pd.DataFrame, target_dt: pd.Series):
        """
        Train clustering model and compute the cluster results of whole and unique coordinates.
        """
        if type(input_dt) is not pd.DataFrame:
            raise Exception("Input data type is not pd.DataFrame.")
        if type(target_dt) is not pd.Series:
            raise Exception("Target data type is not pd.Series.")
        if not input_dt.index.equals(target_dt.index):
            raise Exception("Input and Output indexes are not equal.")
        if "cmaq_x" not in input_dt.columns:
            raise Exception("'cmaq_x' must be in the input data columns.")
        if "cmaq_y" not in input_dt.columns:
            raise Exception("'cmaq_y' must be in the input data columns.")

        return self._cluster_coords(input_dt[["cmaq_x", "cmaq_y", "cmaq_id"]], self.cluster_method, self.cluster_num)

    def split_train_test(self, input_dt: pd.DataFrame, whole_cluster: pd.DataFrame):
        """
        Split train-test data and coordinates.
        """
        if type(input_dt) is not pd.DataFrame:
            raise Exception("Input data type is not pd.DataFrame.")
        if type(whole_cluster) is not pd.DataFrame:
            raise Exception("Whole Cluster data type is not pd.DataFrame.")

        xy_cluster = input_dt[["cmaq_x", "cmaq_y"]].join(whole_cluster)
        return _split_train_test(xy_cluster)
