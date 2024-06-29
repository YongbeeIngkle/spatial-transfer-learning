from data_process.data_path import predict_result_path
from data_process.pred_data import LimaPredLdfSet
from model.regressor import LimaPredictLdfCompose, save_pred
from model.adapt import GbrTrainTest

def _compose_input_label(s_type, f_name, near_station_number):
    feature_compute = LimaPredictLdfCompose(s_type, near_station_number, ldf_a)
    if f_name == "NF":
        all_features = None
    elif f_name == "LDF":
        source_feature, train_target_feature, valid_feature = feature_compute.compute_train_ldf(False)
        all_features = {"source": source_feature, "train_target": train_target_feature, "valid": valid_feature}
    source_set, train_target_set, _ = feature_compute.combine_train_input_feature(all_features)
    return source_set, train_target_set

def _train_predict(source_type, f_name, near_station_num):
    source_set, train_target_set = _compose_input_label(source_type, f_name, near_station_num)
    model_train_test = GbrTrainTest(model_name, ldf_a)
    model_train_test.train(source_set, train_target_set)
    pred_ldf_set = LimaPredLdfSet(source_type, near_station_num, f_name, ldf_a)
    for pred_data in pred_ldf_set:
        pred_cmaq = pred_data["coord"]
        try:
            pred_result = model_train_test.predict(pred_data, False)
        except:
            continue
        save_pred(pred_cmaq, pred_result, prediction_folder, pred_data["file"])

if __name__ == "__main__":
    source_type = "whole"
    model_name = "Nnw"
    feature_name = "LDF" ## what type of characeristic feature is to be produced -- SWA, DWA, LDF
    ldf_a = False
    nearest_station_num = 12
    result_path = predict_result_path["lima"]

    if ldf_a:
        prediction_folder = f"{result_path}/{model_name}/{feature_name}/{source_type} ldf-a/"
    else:
        prediction_folder = f"{result_path}/{model_name}/{feature_name}/{source_type}/"
    _train_predict(source_type, feature_name, nearest_station_num)
