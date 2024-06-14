from data_process.split import LimaSplitLdfInputCompose

if __name__ == "__main__":
    target_train_numbers = [6] ## 6
    source_types = ["whole"] ## other source data options -- "west", "east-north"
    number_of_splits = 5
    station_num = 12
    ldf_a = False

    for train_num in target_train_numbers:
        for s_type in source_types:
            split_compose = LimaSplitLdfInputCompose(train_num, station_num, ldf_a)
            split_compose.save(number_of_splits, s_type)
