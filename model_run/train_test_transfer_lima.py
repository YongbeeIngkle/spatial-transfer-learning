from model.regressor import LimaSplitLdfCompose
from model.adapt import GbrTrainTest, NnFeatureTrainTest, NnParameterTrainTest, algorithm_class
from model.classic import TrainTestTransferData
from result.save import save_accuracy

def _compose_input_label(s_type, f_name, near_station_number, t_num, split_id):
    feature_compute = LimaSplitLdfCompose(s_type, near_station_number, t_num, ldf_a)
    if f_name == "NF":
        all_features = None
    elif f_name == "LDF":
        source_feature, train_target_feature, valid_feature = feature_compute.compute_ldf(split_id, ldf_train)
        all_features = {"source": source_feature, "train_target": train_target_feature, "valid": valid_feature}
    source_set, train_target_set, valid_set = feature_compute.combine_input_feature(split_id, all_features)
    return source_set, train_target_set, valid_set

if __name__ == "__main__":
    source_type = "whole" ## The area of source -- east, west, east-north
    model_name = "Nnw"
    feature_name = "LDF" ## what type of characeristic feature is to be produced -- LDF, NF
    ldf_a = True

    ldf_train = False
    near_station_number = 12
    train_target_number = 6
    number_of_split = 1

    accuracy_file = f"lima {model_name} {feature_name} {source_type}"
    all_label, all_pred, all_mapie = {}, {}, {}
    for split_id in range(number_of_split):
        print(f"split{split_id} train-test")
        source_set, train_target_set, valid_set = _compose_input_label(source_type, feature_name, near_station_number, train_target_number, split_id)
        if model_name in algorithm_class["gbr"]:
            model_train_test = GbrTrainTest(model_name, ldf_a)
        elif model_name in algorithm_class["nn_feature"]:
            model_train_test = NnFeatureTrainTest(model_name)
        elif model_name in algorithm_class["nn_parameter"]:
            model_train_test = NnParameterTrainTest(model_name)
        elif model_name == "GBR":
            model_train_test = TrainTestTransferData(model_name)
        model_train_test.train(source_set, train_target_set)
        split_pred, split_label = model_train_test.predict(valid_set)
        all_label[split_id], all_pred[split_id] = split_label, split_pred
    save_accuracy(all_label, all_pred, accuracy_file, train_target_number)
