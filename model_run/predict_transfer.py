from data_process.lima_compose import PredLdfSet
from model.regressor import PredictFeatureCompute
from model.adapt import GbrTrainTest
from result.save import save_pred

def _compose_input_label(country_name, f_name, station_num):
    feature_compute = PredictFeatureCompute(country_name, source_type, station_num, f_name, ldf_a)
    if f_name == "NF":
        all_features = None
    elif f_name == "LDF":
        source_feature, train_target_feature, valid_feature = feature_compute.compute_ldf(ldf_train)
        all_features = {"source": source_feature, "train_target": train_target_feature, "valid": valid_feature}
    source_set, train_target_set, valid_set = feature_compute.combine_input_feature(all_features)
    return source_set, train_target_set, valid_set

def _train_predict(country_name, f_name, station_num):
    source_set, train_target_set, _ = _compose_input_label(country_name, f_name, station_num)
    model_train_test = GbrTrainTest(model_name)
    model_train_test.train(source_set, train_target_set)
    pred_ldf_set = PredLdfSet(country_name, source_type, station_num, f_name, ldf_a)
    for pred_data in pred_ldf_set:
        pred_cmaq = pred_data["coord"]
        pred_result = model_train_test.predict(pred_data, False)
        save_pred(pred_cmaq, pred_result, prediction_folder, pred_data["file"])

if __name__ == "__main__":
    country_name = "lima"
    source_type = "whole"
    model_name = "Nnw"
    feature_names = ["LDF"] ## what type of characeristic feature is to be produced -- SWA, DWA, LDF
    ldf_a = False
    nearest_station_num = 12
    ldf_train = True
    
    for f_name in feature_names:
        prediction_folder = f"D:/US_PM25_data/{country_name}/predictions/{model_name}/{f_name}/"
        _train_predict(country_name, f_name, nearest_station_num)
