from model.regressor import SplitFeatureCompute
import pandas as pd
import matplotlib.pyplot as plt
from data_process.tag_info import input_tags

def _compose_input_label(s_type, near_num, t_num, input_normalize, split_id=0):
    feature_compute = SplitFeatureCompute(
        s_type, near_num, "LDF", t_num, similarity_measure, 
        coord_pm_input, input_weighting, contextual_input
        )
    source_feature, train_target_feature, valid_feature = feature_compute.compute_ldf(split_id, ldf_train)
    all_features = {"source": source_feature, "train_target": train_target_feature, "valid": valid_feature}
    source_set, train_target_set, valid_set = feature_compute.combine_input_feature(split_id, all_features, input_normalize)
    feature_tags = input_tags + ["LDF"]
    time_exclude_tags = [t for t in feature_tags if t not in ["day", "month"]]
    all_data_df = pd.DataFrame(valid_set["input"], columns=feature_tags)
    all_data_df["PM25"] = valid_set["label"]
    label_corr_values = all_data_df.corr()["PM25"][time_exclude_tags]
    top5_corr = label_corr_values.abs().sort_values(ascending=False)[:5]
    tag_show_names = [t.split("_")[-1] for t in top5_corr.index]
    ax = plt.subplot(111)
    ax.bar(tag_show_names, top5_corr)
    plt.xlabel("Features", fontsize=13)
    plt.ylabel("Correlation to PM2.5 Variable", fontsize=13)
    ax.spines[['right', 'top']].set_visible(False)
    plt.show()

if __name__ == "__main__":
    source_types = ["east", "west", "east-north"] ## The area of source -- east, west, east-north
    coord_pm_input = False
    input_normalize = True

    ldf_train = False
    feature_only = False
    near_station_numbers = [12]
    train_numbers = [9] ### 3,5,7,9,11,13,15
    similarity_measure = "n_dim_eu_distance" ## How neighborhood is calculated -- "eu_distance", "n_dim_eu_distance", "n_dim_cos_distance", "spatial_similarity", "optimal_neighbor", "haver_distance"
    number_of_split = 20
    mapie_saving = True

    for s_type in source_types:
        for near_num in near_station_numbers:
            for contextual_input in [False]: ### To perform contextual LDF makes this True else False
                for input_weighting in [True]:
                    for t_num in train_numbers:
                        all_label, all_pred, all_mapie = {}, {}, {}
                        for split_id in range(number_of_split):
                            print(f"split{split_id} train-test")
                            _compose_input_label(s_type, near_num, t_num, input_normalize)
