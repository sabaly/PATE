def basic_partition(train_features, train_labels, nb_tchrs):
    assert len(train_features) == len(train_labels)

    batch_size = len(train_features)//nb_tchrs
    subsets = [] 
    for i in range(nb_tchrs):
        tchr_features = train_features[i*batch_size:(i+1)*batch_size]
        tchr_labels = train_labels[i*batch_size:(i+1)*batch_size]
        subsets.append((tchr_features, tchr_labels))
    return subsets
