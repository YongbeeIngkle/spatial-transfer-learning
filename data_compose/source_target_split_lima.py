from data_process.split import LimaSourceTargetId

if __name__=='__main__':
    target_train_numbers = [7]
    number_of_splits = 5

    source_target_id = LimaSourceTargetId(target_train_numbers)
    source_target_id.save(number_of_splits, 100)
