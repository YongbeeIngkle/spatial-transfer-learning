from model.regressor import CountrySplitFeatureCompute
from model.adapt import GbrTrainTest, NnFeatureTrainTest, NnParameterTrainTest, algorithm_class
from model.classic import TrainTestTransferData
from result.save import save_accuracy, save_mapie

def _compose_input_label(s_type, f_name, near_num, t_num, input_normalize, split_id):
    if f_name == "NF":
        feature_compute = CountrySplitFeatureCompute(
            "lima", s_type, near_num, f_name, t_num, similarity_measure, 
            False, False, False
            )
        all_features = None
    else:
        if f_name == "LDF":
            feature_compute = CountrySplitFeatureCompute(
                "lima", s_type, near_num, f_name, t_num, similarity_measure, 
                coord_pm_input, input_weighting, contextual_input
                )
            source_feature, train_target_feature, valid_feature = feature_compute.compute_ldf(split_id, ldf_train)
        elif f_name == "DWA":
            feature_compute = SplitFeatureCompute(
                s_type, near_num, f_name, t_num, "eu_distance", 
                False, True, False
                )
            source_feature, train_target_feature, valid_feature = feature_compute.compute_wa(split_id)
        elif f_name == "SWA":
            feature_compute = SplitFeatureCompute(
                s_type, near_num, f_name, t_num, "spatial_similarity", 
                False, True, False
                )
            source_feature, train_target_feature, valid_feature = feature_compute.compute_wa(split_id)
        all_features = {"source": source_feature, "train_target": train_target_feature, "valid": valid_feature}
    source_set, train_target_set, valid_set = feature_compute.combine_input_feature(split_id, all_features, input_normalize)
    if feature_only:
        source_set["input"] = source_set["input"][:,-1].reshape(-1,1)
        train_target_set["input"] = train_target_set["input"][:,-1].reshape(-1,1)
        valid_set["input"] = valid_set["input"][:,-1].reshape(-1,1)
    return source_set, train_target_set, valid_set

if __name__ == "__main__":
    source_types = ["whole"] ## The area of source -- east, west, east-north
    model_name = "Nnw"
    feature_names = ["NF", "LDF"] ## what type of characeristic feature is to be produced -- SWA, DWA, LDF, NF
    coord_pm_input = False
    input_normalize = True

    ldf_train = True
    feature_only = False
    near_station_numbers = [12]
    train_numbers = [6] ### 3,5,7,9,11,13,15
    similarity_measure = "n_dim_eu_distance" ## How neighborhood is calculated -- "eu_distance", "n_dim_eu_distance", "n_dim_cos_distance", "spatial_similarity", "optimal_neighbor", "haver_distance"
    number_of_split = 5
    mapie_saving = False

    for s_type in source_types:
        for f_name in feature_names:
            for near_num in near_station_numbers:
                for contextual_input in [True, False]: ### To perform contextual LDF makes this True else False
                    for input_weighting in [True]:
                        accuracy_file = f"lima {model_name} {f_name} {s_type} {similarity_measure}"
                        if f_name == "LDF":
                            if coord_pm_input:
                                accuracy_file += " coord_pm"
                            if input_weighting:
                                accuracy_file += " input_weight"
                            if contextual_input:
                                accuracy_file += " contextual_input"
                            if feature_only:
                                accuracy_file += " feature_only"
                        elif contextual_input:
                            continue
                        for t_num in train_numbers:
                            all_label, all_pred, all_mapie = {}, {}, {}
                            for split_id in range(number_of_split):
                                print(f"split{split_id} train-test")
                                source_set, train_target_set, valid_set = _compose_input_label(s_type, f_name, near_num, t_num, input_normalize, split_id)
                                if model_name in algorithm_class["gbr"]:
                                    model_train_test = GbrTrainTest(model_name)
                                elif model_name in algorithm_class["nn_feature"]:
                                    model_train_test = NnFeatureTrainTest(model_name)
                                elif model_name in algorithm_class["nn_parameter"]:
                                    model_train_test = NnParameterTrainTest(model_name)
                                elif model_name == "GBR":
                                    model_train_test = TrainTestTransferData(model_name)
                                model_train_test.train(source_set, train_target_set)
                                if mapie_saving:
                                    all_mapie[split_id] = model_train_test.mapie(source_set, train_target_set, valid_set)
                                else:
                                    split_pred, split_label = model_train_test.predict(valid_set)
                                    all_label[split_id], all_pred[split_id] = split_label, split_pred
                            if mapie_saving:
                                save_mapie(all_mapie)
                            else:
                                save_accuracy(all_label, all_pred, accuracy_file, t_num)
