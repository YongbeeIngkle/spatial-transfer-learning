from data_process.data_path import predict_result_path
from data_process.pred_data import CaliforniaPredLdfSet
from model.regressor import CaliforniaSplitLdfCompose, save_pred
from model.adapt import GbrTrainTest

def _compose_input_label(s_type, f_name, near_station_number, t_num, split_id):
    feature_compute = CaliforniaSplitLdfCompose(s_type, near_station_number, t_num, ldf_a)
    if f_name == "NF":
        all_features = None
    elif f_name == "LDF":
        source_feature, train_target_feature, valid_feature = feature_compute.compute_ldf(split_id, False)
        all_features = {"source": source_feature, "train_target": train_target_feature, "valid": valid_feature}
    source_set, train_target_set, _ = feature_compute.combine_input_feature(split_id, all_features)
    return source_set, train_target_set

def _train_predict(source_type, target_station_num, f_name, near_station_num, split_id):
    source_set, train_target_set = _compose_input_label(source_type, f_name, near_station_num, target_station_num, split_id)
    model_train_test = GbrTrainTest(model_name, ldf_a)
    model_train_test.train(source_set, train_target_set)
    pred_ldf_set = CaliforniaPredLdfSet(target_station_num, source_type, near_station_num, f_name, split_id, ldf_a)
    for pred_data in pred_ldf_set:
        pred_cmaq = pred_data["coord"]
        pred_result = model_train_test.predict(pred_data, False)
        save_pred(pred_cmaq, pred_result, prediction_folder, pred_data["file"])

if __name__ == "__main__":
    source_type = "east-north"
    model_name = "Nnw"
    feature_name = "LDF" ## what type of characeristic feature is to be produced -- SWA, DWA, LDF
    ldf_a = False
    target_station_num = 9
    nearest_station_num = 12
    split_id = 0
    result_path = predict_result_path["california"]

    prediction_folder = f"{result_path}/{model_name}/{feature_name}/"
    _train_predict(source_type, target_station_num, feature_name, nearest_station_num, split_id)
