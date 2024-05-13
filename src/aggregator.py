import numpy as np

teachers = []
def init_teachers(tchrs):
    global teachers
    teachers = tchrs

fairness_metrics = []
def set_metrics(metrics):
    global fairness_metrics
    fairness_metrics = metrics

def laplacian_noisy_vote(data_to_label, gamma=0.1, voters=[]):
    # predictions from teachers
    if voters == []:
        voters = teachers.copy()
    preds = []
    for tchr in teachers:
        pred = tchr.predict(data_to_label)
        pred = np.round(pred)
        preds.append([p[0] for p in pred])
    preds = np.asarray(preds, dtype=np.int32)
    # voting
    labels = []
    for x in range(np.shape(preds)[1]):
        n_y_x = np.bincount(preds[:,x])
        # adding noise to votes
        for i in range(len(n_y_x)):
            n_y_x[i] += np.random.laplace(loc=0.0, scale=gamma)
        labels.append(np.argmax(n_y_x))
    return labels

def gaussian_noisy_vote(data_to_label, mu=0, sigma=1, voters=[]):
    # predictions from teachers
    if voters == []:
        voters = teachers.copy()
    preds = []
    for tchr in voters:
        pred = tchr.predict(data_to_label)
        pred = np.round(pred)
        preds.append([p[0] for p in pred])
    preds = np.asarray(preds, dtype=np.int32)
    # voting
    labels = []
    for x in range(np.shape(preds)[1]):
        n_y_x = np.bincount(preds[:,x])
        # adding noise to votes
    
        n_y_x = n_y_x + np.random.normal(mu, sigma, len(n_y_x))
        labels.append(np.argmax(n_y_x))
    return labels

def plurality(data_to_label, voters=[]):
    # predictions from teachers
    if voters == []:
        voters = teachers.copy()
    preds = []
    for tchr in teachers:
        pred = tchr.predict(data_to_label)
        pred = np.round(pred)
        preds.append([p[0] for p in pred])
    preds = np.asarray(preds, dtype=np.int32)
    # voting
    labels = []
    for x in range(np.shape(preds)[1]):
        n_y_x = np.bincount(preds[:,x])
        labels.append(np.argmax(n_y_x))
    return labels

methode = plurality
def only_fair(data_to_label):
    global teachers
    to_ban = []
    for i in range(len(teachers)):
        if fairness_metrics[i] > 0.05:
            to_ban.append(teachers[i])
    voters = list(set(teachers) - set(to_ban))
    return methode(data_to_label, voters=voters)




def update_aggregator(current):
    choice = input("1. Plurality \t 2. LNMax\t 3. GNMax \t 4. Only Fair\n (0 to exit)>>> ")
    if choice == "1":
        return plurality
    elif choice == "2":
        return laplacian_noisy_vote
    elif choice == "3":
        return gaussian_noisy_vote
    elif choice == "4":
        global methode
        choice = input("Methode : 1. Plurality \t 2. LNMax\t 3. GNMax\n>>> ")
        if choice == "3":
            methode = gaussian_noisy_vote
        elif choice == "2":
            methode = laplacian_noisy_vote
        else:
            methode = plurality
           
        print(f"Only Fair teachers participates using {methode.__name__}")
        return only_fair
    else:
        return current