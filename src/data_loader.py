import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from art.utils import load_mnist
import numpy as np
import pandas as pd
from partition import *
from random import randint

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

    return X, Y, S 


states = ["HI", "CA", "AK", "PR", "NV", "NM", "OK", "NY", "WA", "AZ",  "MD",
"TX", "VA", "MA", "GA", "CT", "OR", "IL", "RI", "NC", "CO", "DE", "LA", "UT",
"FL", "MS", "SC", "AR", "SD", "AL", "MI", "KS", "ID", "MN", "MT", "OH", "IN",
"TN", "PA", "NE", "MO", "WY", "ND", "WI", "KY", "NH", "ME", "IA", "VT", "WV"] # "NJ"

alpha = [150,100]
uf_alphas = [[20, 50], [50, 100], [300, 5], [50, 100], [50, 100], [50, 100], [350, 10], [50, 100], [20, 50], [200, 10], [50, 100], [250, 1], [50, 100], [250, 20], [250, 1], [50, 100], [300, 5], [50, 100], [300, 5], [20, 50], [20, 50], [250, 20], [50, 100], [50, 100], [250, 20], [200, 10], [50, 100], [20, 50], [250, 20], [20, 50]]
def update_alpha(new_alpha):
    global alpha
    alpha = new_alpha


def load_ACSEmployment(year=2018, horizon="1-Year", states=states, nb_fair_tchrs=0):
    data_src = ACSDataSource(survey_year=year, horizon=horizon, survey="person")
    subsets = []
    if len(states) > 2:
        states.pop(2) # delete student
    for st in states:
        acs_data = data_src.get_data(states=[st], download=True)
        features, labels, group = ACSEmployment.df_to_numpy(acs_data)

        x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(
            features, labels, group, test_size=0.2, random_state=0
        )
        subsets.append((x_train, x_test, y_train, y_test, s_train, s_test))
    return subsets

    
def load_ACSEmployment_bis(year=2018, horizon="1-Year", states=states, nb_fair_tchrs=0):
    data_src = ACSDataSource(survey_year=year, horizon=horizon, survey="person")
    subsets = []
    if len(states) > 2:
        states.pop(2) # delete student
    fair = 0
    for st in states:
        acs_data = data_src.get_data(states=[st], download=True)
        features, labels, group = ACSEmployment.df_to_numpy(acs_data)

        if fair < nb_fair_tchrs:
            fair += 1
        else:
            df = pd.DataFrame(features)
            df.columns = ACSEmployment.features
            df[ACSEmployment.target] = labels

            p_grp_pr = df[(df["RAC1P"] == 1) & (df["ESR"] == True)]
            up_grp_pr = df[(df["RAC1P"] == 2) & (df["ESR"] == True)]
            rest_of_df = df[((df["RAC1P"] != 1) & (df["RAC1P"] != 2)) | (df["ESR"] == False)]
            p_vs_up = pd.concat([p_grp_pr, up_grp_pr])
            alpha = uf_alphas[states.index(st)]
            dist = np.random.dirichlet(alpha, 1)
            size_p_grp = int(dist[0][0]*p_vs_up.shape[0])
            size_up_grp = p_vs_up.shape[0]-size_p_grp

            p_grp = p_grp_pr.sample(size_p_grp, replace=True)
            up_grp = up_grp_pr.sample(size_up_grp, replace=True)
            final_df = pd.concat([p_grp, up_grp, rest_of_df])

            labels = np.array(final_df.pop("ESR"))
            features = final_df.copy()
            group = final_df["RAC1P"]

        x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(
            features, labels, group, test_size=0.2, random_state=0
        )
        subsets.append((x_train, x_test, y_train, y_test, s_train, s_test))
    return subsets

def load_student_data(state, year=2018, horizon="1-Year"):
    data_src = ACSDataSource(survey_year=year, horizon=horizon, survey="person")
    acs_data = data_src.get_data(states=[state], download=True)
    features, labels, group = ACSEmployment.df_to_numpy(acs_data)
    x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(
            features, labels, group, test_size=0.2, random_state=0
        )
    return (x_train, x_test, y_train, y_test, s_train, s_test)

def get(dataset_name, nb_teachers=49, nb_fair_tchrs=0):
    if dataset_name == "adult":
        X,Y,S = clear_adult_data(ld_adult())
        subsets = adult_basic_partition(X, Y, S, nb_teachers)
        return subsets, None
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
    elif dataset_name == "acsemployment":
        return load_ACSEmployment(states=states[:nb_teachers+1], nb_fair_tchrs=nb_fair_tchrs), states[2]
    elif dataset_name == "acsemployment_bis":
        return load_ACSEmployment_bis(states=states[:nb_teachers+1], nb_fair_tchrs=nb_fair_tchrs), states[2]
    else:
        return None


