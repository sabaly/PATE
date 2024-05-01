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

def ld_compas():
    pass

def ld_mnist():
    pass


