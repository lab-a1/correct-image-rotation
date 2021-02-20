def split_dataset(dataset, validation_size, test_size):
    validation_dataset_size = int(len(dataset) * validation_size)
    test_dataset_size = int(len(dataset) * test_size)

    validation_dataset = dataset[:validation_dataset_size]
    validation_dataset.reset_index(inplace=True, drop=True)
    test_dataset = dataset[
        validation_dataset_size : validation_dataset_size + test_dataset_size
    ]
    test_dataset.reset_index(inplace=True, drop=True)
    train_dataset = dataset[validation_dataset_size + test_dataset_size :]
    train_dataset.reset_index(inplace=True, drop=True)

    return train_dataset, validation_dataset, test_dataset
