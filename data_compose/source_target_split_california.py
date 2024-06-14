from data_process.split import CaliforniaSourcetargetId

if __name__=='__main__':
    cluster_id = 6
    target_train_numbers = [5,7,9,11]
    number_of_splits = 20

    source_target_id = CaliforniaSourcetargetId(cluster_id, target_train_numbers)
    source_target_id.save(number_of_splits)
