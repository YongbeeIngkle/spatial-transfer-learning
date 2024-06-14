import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import haversine_distances

def create_distance_matrix(dt1: pd.DataFrame, dt2: pd.DataFrame):
    """
    Create distance matrix between two dataframe. If the distance is 0 (i.e. same coordinate), replace to inf.

    dt1: first data
    dt2: second data
    """

    all_distance = distance_matrix(dt1, dt2)
    all_distance[all_distance==0] = np.inf
    all_distance = pd.DataFrame(all_distance, index=dt1.index, columns=dt2.index)
    return all_distance

def create_distance_pairwise(dt1: np.ndarray, dt2: np.ndarray):
    all_distance = np.linalg.norm(dt1 - dt2, axis=1)
    return all_distance

def create_cosine_matrix(dt1: pd.DataFrame, dt2: pd.DataFrame):
    """
    Create cosine distance matrix between two dataframe. If the distance is 0 (i.e. same coordinate), replace to inf.

    dt1: first data
    dt2: second data
    """
    all_distance = cdist(dt1, dt2, metric='cosine') ##Similarity = 1 - Distance :: Hence, less cosine distance => More similarity
    all_distance[all_distance==0] = np.inf
    all_distance = pd.DataFrame(all_distance, index=dt1.index, columns=dt2.index)
    return all_distance

def create_cosine_pairwise(dt1: np.ndarray, dt2: np.ndarray):
    # Compute dot product
    dot_product = np.sum(dt1 * dt2, axis=1)
    
    # Compute magnitudes
    magnitude_arr1 = np.linalg.norm(dt1, axis=1)
    magnitude_arr2 = np.linalg.norm(dt2, axis=1)
    
    # Compute cosine similarity
    cosine_sim = dot_product / (magnitude_arr1 * magnitude_arr2)
    
    # Compute cosine distance
    cosine_dist = 1 - cosine_sim
    return cosine_dist

def create_haversine_matrix(dt1: pd.DataFrame, dt2: pd.DataFrame):
    """
    Create distance matrix between two dataframes using Haversine distance.
    If the distance is 0 (i.e. same coordinate), replace it with inf.

    dt1: first dataframe with latitude and longitude columns
    dt2: second dataframe with latitude and longitude columns
    """
    # Convert latitude and longitude from degrees to radians
    df1_rad = np.radians(dt1[['cmaq_x', 'cmaq_y']].values)
    df2_rad = np.radians(dt2[['cmaq_x', 'cmaq_y']].values)

    # Compute haversine distances
    distances = haversine_distances(df1_rad, df2_rad) * 6371  # Radius of Earth in kilometers
    distances = pd.DataFrame(distances, index=dt1.index, columns=dt2.index)
    return distances

def create_haversine_pairwise(dt1: np.ndarray, dt2: np.ndarray):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = np.radians(dt1['cmaq_y']), np.radians(dt1['cmaq_x'])
    lat2, lon2 = np.radians(dt2['cmaq_y']), np.radians(dt2['cmaq_x'])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Calculate the distance
    distances = R * c
    return distances

def compute_distance_matrix(similarity_measure: str, compute_input: pd.DataFrame, train_input: pd.DataFrame):
    if "pm25_value" in compute_input.columns:
        Exception("Pm2.5 value must not be in the distance measuring tags.")
    if similarity_measure == "eu_distance":
        distance = create_distance_matrix(compute_input, train_input)
    elif similarity_measure == "n_dim_eu_distance":
        distance = create_distance_matrix(compute_input, train_input)
    elif similarity_measure == "n_dim_cos_distance":
        distance = create_cosine_matrix(compute_input, train_input)
    elif similarity_measure == "spatial_similarity":
        distance = create_distance_matrix(compute_input, train_input)
    elif similarity_measure == "haver_distance":
        distance = create_haversine_matrix(compute_input, train_input)
    return distance

def compute_distance_pairwise(similarity_measure: str, compute_input: np.ndarray, train_input: np.ndarray):
    if similarity_measure == "eu_distance":
        distance = create_distance_pairwise(compute_input, train_input)
    elif similarity_measure == "n_dim_eu_distance":
        distance = create_distance_pairwise(compute_input, train_input)
    elif similarity_measure == "n_dim_cos_distance":
        distance = create_cosine_pairwise(compute_input, train_input)
    elif similarity_measure == "spatial_similarity":
        distance = create_distance_pairwise(compute_input, train_input)
    elif similarity_measure == "optimal_neighbor":
        distance = create_distance_pairwise(compute_input, train_input)
    elif similarity_measure == "haver_distance":
        distance = create_haversine_pairwise(compute_input, train_input)
    return distance
