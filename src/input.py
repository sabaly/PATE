import sklearn
from sklearn.preprocessing import LabelEncoder

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
    labels = [1 if x == ">50K" else 0 for x in labels]

    le = LabelEncoder()
    features['education'] = le.fit_transform(features['education'])
    features['marital-status'] = le.fit_transform(features['marital-status'])
    features['occupation'] = le.fit_transform(features['occupation'])
    features['relationship'] = le.fit_transform(features['relationship'])
    features['native-country'] = le.fit_transform(features['native-country'])
    
    return (features, labels)

""" MNIST dataset """
def ld_mnist():
    pass


