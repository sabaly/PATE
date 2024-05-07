import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from art.utils import load_mnist
import numpy as np

from folktables import ACSDataSource, ACSEmployment

""" Adult dataset """
def ld_adult():
    # make sure you've installed the repo : pip install ucimlrepo
    from ucimlrepo import fetch_ucirepo 

    # fetch dataset
    print("fetching dataset...", end="")
    adult = fetch_ucirepo(id=2)
    print("Done")
      
    # data (as pandas dataframes) 
    #X = adult.data.features 
    #y = adult.data.targets
    return adult

def clear_adult_data(adult):
    features = adult.data.features
    labels = adult.data.targets
    features = features.drop(['workclass', 'fnlwgt'], axis=1)
    labels = [0 if x == "<=50K" else 1 for x in labels['income']]

    le = LabelEncoder()
    features['education'] = le.fit_transform(features['education'])
    features['marital-status'] = le.fit_transform(features['marital-status'])
    features['occupation'] = le.fit_transform(features['occupation'])
    features['relationship'] = le.fit_transform(features['relationship'])
    features['native-country'] = le.fit_transform(features['native-country'])
    features['sex'] = [1 if x=='Male' else 2 for x in features['sex']]
    features['race'] = [1 if x=='White' else 2 for x in features['sex']]

    Y = labels.copy()
    X = features.copy()
    S = features['sex'].copy()
    X_train, X_test, Y_train , Y_test, S_train, S_test = train_test_split(X, Y, S, test_size=0.2)

    x_train = np.array(X_train)
    y_train = np.array(Y_train)
    s_train = np.array(S_train)
    s_test = np.array(S_test)

    x_test = np.array(X_test)
    y_test = np.array(Y_test)
    return x_train, x_test, y_train, y_test, s_train, s_test

def load_ACSDataSource(year=2015, horizon="1-Year"):
    return ACSDataSource(survey_year=year, horizon=horizon, survey="person")


def get(dataset_name):
    if dataset_name == "adult":
        return clear_adult_data(ld_adult())
    elif dataset_name == "mnist":
        (x_train, train_label), (x_test, test_label), _, _ = load_mnist()
        y_train = []
        for i in range(len(train_label)):
            y_train.append(int(np.argmax(train_label[i])))
        y_test = []
        for i in range(len(test_label)):
            y_test.append(int(np.argmax(test_label[i])))
        y_train, y_test = np.array(y_train), np.array(y_test)
        return x_train, x_test, y_train, y_test
    else:
        return None


