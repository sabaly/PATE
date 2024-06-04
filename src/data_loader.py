#import sklearn
#from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from random import choice
from folktables import ACSDataSource, ACSEmployment

states = ["HI", "CA", "PR", "NV", "NM", "OK", "NY", "WA", "AZ",  "MD",
"TX", "VA", "MA", "GA", "CT", "OR", "IL", "RI", "NC", "CO", "DE", "LA", "UT",
"FL", "MS", "SC", "AR", "SD", "AL", "MI", "KS", "ID", "MN", "MT", "OH", "IN",
"TN", "PA", "NE", "MO", "WY", "ND", "WI", "KY", "NH", "ME", "IA", "VT", "WV"] # "NJ" can't be download and "AK" is the student dataset


data_src = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")

alpha = [100,100]
alphas = [[100, 100]]*(len(states) - 1)
alphas[9] = [100,180]
def update_alpha(new_alpha):
    global alpha
    alpha = new_alpha

def load_ACSEmployment(year=2018, horizon="1-Year", states=states):
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
    #data_src = ACSDataSource(survey_year=year, horizon=horizon, survey="person")
    subsets = []
    fair_st = []
    if nb_fair_tchrs < len(states):
        for _ in range(nb_fair_tchrs):
            fair_st.append(choice(states))
    else:
        fair_st = states.copy()
    teachers_s = []
    id = 0
    for st in states:
        acs_data = data_src.get_data(states=[st], download=True)
        features, labels, group = ACSEmployment.df_to_numpy(acs_data)
        #teachers_s.append(aggregate_dataset(id), features, labels, group)
        id += 1
        if st not in fair_st:
            df = pd.DataFrame(features)
            df.columns = ACSEmployment.features
            df[ACSEmployment.target] = labels

            p_grp_pr = df[(df["RAC1P"] == 1) & (df["ESR"] == True)]
            up_grp_pr = df[(df["RAC1P"] == 2) & (df["ESR"] == True)]
            rest_of_df = df[((df["RAC1P"] != 1) & (df["RAC1P"] != 2)) | (df["ESR"] == False)]
            p_vs_up = pd.concat([p_grp_pr, up_grp_pr])
            alpha = alphas[states.index(st)]
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
    S = pd.concat(teachers_s)
    return subsets, S

def load_student_data(state, year=2018, horizon="1-Year"):
    data_src = ACSDataSource(survey_year=year, horizon=horizon, survey="person")
    acs_data = data_src.get_data(states=[state], download=True)
    features, labels, group = ACSEmployment.df_to_numpy(acs_data)
    x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(
            features, labels, group, test_size=0.2, random_state=0
        )
    return (x_train, x_test, y_train, y_test, s_train, s_test)

def get(dataset_name, nb_teachers=49, nb_fair_tchrs=0):
    if dataset_name == "acsemployment":
        return load_ACSEmployment(states=states[:nb_teachers+1]), states[2]
    elif dataset_name == "acsemployment_bis":
        return load_ACSEmployment_bis(states=states[:nb_teachers+1], nb_fair_tchrs=nb_fair_tchrs)
    else:
        return None

