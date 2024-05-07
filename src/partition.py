
from sklearn.model_selection import train_test_split
import numpy as np

def basic_partition(train_features, train_labels, test_features, test_labels, 
s_features, s_labels, nb_tchrs):
    assert len(train_features) == len(train_labels)

    train_batch_size = len(train_features)//nb_tchrs
    test_batch_size = len(test_features) // nb_tchrs
    s_batch_size = len(s_features) // nb_tchrs
    subsets = [] 
    for i in range(nb_tchrs):
        x_train = train_features[i*train_batch_size:(i+1)*train_batch_size]
        y_train = train_labels[i*train_batch_size:(i+1)*train_batch_size]
        x_test = test_features[i*test_batch_size:(i+1)*test_batch_size]
        y_test = test_labels[i*test_batch_size:(i+1)*test_batch_size]
        s_train = s_features[i*s_batch_size:(i+1)*s_batch_size]
        s_test = s_labels[i*s_batch_size:(i+1)*s_batch_size]
        subsets.append((x_train, y_train, x_test, y_test, s_train, s_test))
    return subsets

def adult_basic_partition(X, Y, S, nb_classes):
    assert len(X) == len(Y)

    batch_size = len(X)//nb_classes

    subsets = []
    for i in range(nb_classes):
        x = X[i*batch_size:(i+1)*batch_size]
        y = Y[i*batch_size:(i+1)*batch_size]
        s = S[i*batch_size:(i+1)*batch_size]
        x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(x, y, s, test_size=0.1)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        s_train = np.array(s_train)
        s_test = np.array(s_test)

        x_test = np.array(x_test)
        y_test = np.array(y_test)
        subsets.append((x_train, x_test, y_train, y_test, s_train, s_test))
    return subsets