from data_process.split import CaliforniaSplitLdfInputCompose

if __name__ == "__main__":
    target_train_numbers = [5,7,9,11] ## 5,7,9,11
    source_type = "east-north" ## "east", "east-north"
    ldf_a = False
    number_of_splits = 20
    station_num = 12

    for train_num in target_train_numbers:
        split_compose = CaliforniaSplitLdfInputCompose(train_num, station_num, ldf_a)
        split_compose.save(number_of_splits, source_type)
