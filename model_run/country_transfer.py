from data_process.lima_compose import PredLdfSet
from model.regressor import CountryFeatureCompute
from model.adapt import GbrTrainTest
from result.save import save_pred

def _compose_input_label(country_name, f_name, station_num, input_normalize):
    if f_name == "NF":
        feature_compute = CountryFeatureCompute(
            country_name, source_type, station_num, f_name, similarity_measure, 
            False, False, False
            )
        all_features = None
    else:
        if f_name == "LDF":
            feature_compute = CountryFeatureCompute(
                country_name, source_type, station_num, f_name, similarity_measure, 
                coord_pm_input, input_weighting, contextual_input, ldf_train
                )
            source_feature, train_target_feature, valid_feature = feature_compute.compute_ldf()
        all_features = {"source": source_feature, "train_target": train_target_feature, "valid": valid_feature}
    source_set, train_target_set, valid_set = feature_compute.combine_input_feature(all_features, input_normalize)
    return source_set, train_target_set, valid_set

def _train_predict(country_name, f_name, station_num, input_normalize):
    source_set, train_target_set, _ = _compose_input_label(country_name, f_name, station_num, input_normalize)
    model_train_test = GbrTrainTest(model_name)
    model_train_test.train(source_set, train_target_set)
    pred_ldf_set = PredLdfSet(country_name, source_type, station_num, f_name, similarity_measure, 
                              coord_pm_input, input_weighting, contextual_input, input_normalize)
    for pred_data in pred_ldf_set:
        pred_cmaq = pred_data["coord"]
        pred_result = model_train_test.predict(pred_data, False)
        save_pred(pred_cmaq, pred_result, prediction_folder, pred_data["file"])

if __name__ == "__main__":
    country_name = "lima"
    source_type = "whole"
    model_name = "Nnw"
    feature_names = ["LDF"] ## what type of characeristic feature is to be produced -- SWA, DWA, LDF
    coord_pm_input = False
    input_weighting = True
    contextual_input = True ### To perform contextual LDF makes this True else False
    nearest_station_num = 12
    input_normalize = True

    ldf_train = True
    # feature_only = False
    similarity_measure = "n_dim_eu_distance" ## How neighborhood is calculated -- "eu_distance", "n_dim_eu_distance", "n_dim_cos_distance", "spatial_similarity", "optimal_neighbor", "haver_distance"

    for f_name in feature_names:
        prediction_folder = f"D:/US_PM25_data/{country_name}/predictions/{model_name}/{f_name}/{similarity_measure}/"
        if f_name == "LDF":
            if coord_pm_input:
                prediction_folder += "coord_pm/"
            if input_weighting:
                prediction_folder += "input_weight/"
            if contextual_input:
                prediction_folder += "contextual_input/"
        _train_predict(country_name, f_name, nearest_station_num, input_normalize)
