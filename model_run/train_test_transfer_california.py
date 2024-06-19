from model.regressor import CaliforniaSplitLdfCompose
from model.adapt import GbrTrainTest, NnFeatureTrainTest, NnParameterTrainTest, algorithm_class
from model.classic import TrainTestTransferData
from result.save import save_accuracy

def _compose_input_label(s_type, f_name, near_station_number, t_num, split_id):
    feature_compute = CaliforniaSplitLdfCompose(s_type, near_station_number, t_num, ldf_a)
    if f_name == "NF":
        all_features = None
    elif f_name == "LDF":
        source_feature, train_target_feature, valid_feature = feature_compute.compute_ldf(split_id, ldf_train)
        all_features = {"source": source_feature, "train_target": train_target_feature, "valid": valid_feature}
    source_set, train_target_set, valid_set = feature_compute.combine_input_feature(split_id, all_features)
    return source_set, train_target_set, valid_set

if __name__ == "__main__":
    source_type = "east-north" ## The area of source -- east, east-north
    model_name = "Nnw"
    feature_name = "NF" ## LDF: Latent Dependency Factor, NF: no feature
    ldf_a = False

    ldf_train = False
    near_station_number = 12 # 4, 8, 12, 16
    train_numbers = [5, 7, 9, 11]
    number_of_split = 20

    accuracy_file = f"{model_name} {feature_name} {source_type}"
    if (feature_name == "LDF") and ldf_a:
        accuracy_file += " ldf-a"
    for t_num in train_numbers:
        all_label, all_pred = {}, {}
        for split_id in range(number_of_split):
            print(f"split{split_id} train-test")
            source_set, train_target_set, valid_set = _compose_input_label(source_type, feature_name, near_station_number, t_num, split_id)
            if model_name in algorithm_class["gbr"]:
                model_train_test = GbrTrainTest(model_name)
            elif model_name in algorithm_class["nn_feature"]:
                model_train_test = NnFeatureTrainTest(model_name)
            elif model_name in algorithm_class["nn_parameter"]:
                model_train_test = NnParameterTrainTest(model_name)
            elif model_name == "GBR":
                model_train_test = TrainTestTransferData(model_name)
            model_train_test.train(source_set, train_target_set)
            split_pred, split_label = model_train_test.predict(valid_set)
            all_label[split_id], all_pred[split_id] = split_label, split_pred
        save_accuracy(all_label, all_pred, accuracy_file, t_num)
