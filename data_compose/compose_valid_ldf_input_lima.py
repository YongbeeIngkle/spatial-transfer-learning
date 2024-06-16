from data_process.split import LimaSplitLdfInputCompose

if __name__ == "__main__":
    target_train_numbers = [6] ## 6
    source_type = "whole" ## other source data options -- "west", "east-north"
    ldf_a = True
    number_of_splits = 5
    station_num = 12

    for train_num in target_train_numbers:
        split_compose = LimaSplitLdfInputCompose(train_num, station_num, ldf_a)
        split_compose.save(number_of_splits, source_type)
