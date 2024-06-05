import numpy as np
from analysis import mean

teachers = []
def update_teachers(tchrs):
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
    for tchr in voters:
        pred = tchr.model.predict(data_to_label)
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
        pred = tchr.model.predict(data_to_label)
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

def plurality(data_to_label, group=[], voters=[]):
    # predictions from teachers
    if voters == []:
        voters = teachers.copy()
    preds = []
    for tchr in voters:
        pred = tchr.model.predict(data_to_label)
        pred = np.round(pred)
        preds.append([p[0] for p in pred])
    preds = np.asarray(preds, dtype=np.int32)
    # voting
    labels = []
    consent = []
    for x in range(np.shape(preds)[1]):
        n_y_x = np.bincount(preds[:,x])
        if len(n_y_x) == 1:
            c = 0
        else:
            c = n_y_x[1]/sum(n_y_x)
        label = np.argmax(n_y_x)
        if not label:
            c = 1 - c
        consent.append(c)
        labels.append(label)
    return np.asarray(labels), np.mean(consent)

methode = plurality
def only_fair(data_to_label, group=[]):
    global teachers
    to_ban = []
    for i in range(len(teachers)):
        if fairness_metrics[i] >= 0.1:
            to_ban.append(teachers[i])
    voters = list(set(teachers) - set(to_ban))
    return methode(data_to_label, voters=voters)

def only_unfair(data_to_label, group=[]):
    global teachers
    to_ban = []
    for i in range(len(teachers)):
        if fairness_metrics[i] < 0.1:
            to_ban.append(teachers[i])
    voters = list(set(teachers) - set(to_ban))
    return methode(data_to_label, voters=voters)

def weighed_vote(data_to_label, group=[], voters=[], fairness = fairness_metrics):
    if fairness == []:
        fairness = fairness_metrics.copy()
    # predictions from teachers
    if voters == []:
        voters = teachers.copy()
    preds = []
    for tchr in voters:
        pred = tchr.model.predict(data_to_label)
        pred = list(np.round(pred))
        preds.append([p[0] for p in pred])
    preds = np.asarray(preds, dtype=np.int32)
    # voting
    labels = []
    consent = []
    for x in range(np.shape(preds)[1]):
        pred = list(preds[:,x])
        for i in range(len(pred)):
            if fairness[i] < 0.1:
                pred.append(pred[i])
        n_y_x = np.bincount(pred)
        if len(n_y_x) == 1:
            c = 0
        else:
            c = n_y_x[1]/sum(n_y_x)
        label = np.argmax(n_y_x)
        if not label:
            c = 1 - c
        consent.append(c)
        labels.append(np.argmax(n_y_x))
    return np.asarray(labels), np.mean(consent)

# ######################
# FairFed weighs
# ######################
def computes_weigh(teachers, beta=1, gamma=100):
    # computes global metric
    fg = 0
    sum_ni = 0
    for tchr in teachers:
        fg += tchr.local_m
        nk = tchr.splited_data[1].shape[0]
        sum_ni += nk
    fg = abs(fg)

    ws = [0]*len(teachers)
    for i in range(len(teachers)):
        nk = teachers[i].splited_data[1].shape[0]
        ws[i] = np.exp(-beta*abs(teachers[i].metrics["EOD"] - fg)) * nk/sum_ni
    sum_ws = sum(ws)
    for i in range(len(ws)):
        ws[i] = ws[i]/sum_ws
    for i in range(len(ws)):
        ws[i] = int(np.floor(gamma*ws[i]))
    return ws

def fair_fed_agg(data_to_label, group=[], voters=[], fairness=fairness_metrics):
    # predictions from teachers
    if fairness == []:
        fairness = fairness_metrics.copy()
    if voters == []:
        voters = teachers.copy()
    preds = []
    for tchr in voters:
        pred = tchr.model.predict(data_to_label)
        pred = np.round(pred)
        preds.append([p[0] for p in pred])
    preds = np.asarray(preds, dtype=np.int32)

    # computes weighs
    weighs = computes_weigh(voters)

    labels = []
    for x in range(np.shape(preds)[1]):
        pred = list(preds[:,x])
        for i in range(len(pred)):
            if fairness[i] < 0.1:
                pred = pred + [pred[i]]*weighs[i]
        n_y_x = np.bincount(pred)
        labels.append(np.argmax(n_y_x))
    return np.asarray(labels), weighs

# ######################
# Spd weighs
# ######################
def get_spd_weighs(x_train, preds, group, beta=100, fairness=[]):
    spds = []
    i = 0
    for yhat in preds:
        spd = 1-abs(mean(x_train[(group == 1) & (yhat == 1)]) - mean(x_train[(group == 2) & (yhat == 1)]))
        i+=1
        spds.append(spd)
    
    sum_ws = sum(spds)
    for i in range(len(spds)):
        spds[i] = spds[i]/sum_ws
    for i in range(len(spds)):
        spds[i] = int(np.floor(beta*spds[i]))
    return spds

def get_spd_weighs_2(x_train, preds, group, beta=100, fairness=[]):
    spds = []
    i = 0
    for yhat in preds:
        spd = 1-fairness[i]*abs(mean(x_train[(group == 1) & (yhat == 1)]) - mean(x_train[(group == 2) & (yhat == 1)]))
        i+=1
        spds.append(spd)
    
    sum_ws = sum(spds)
    for i in range(len(spds)):
        spds[i] = spds[i]/sum_ws
    for i in range(len(spds)):
        spds[i] = int(np.floor(beta*spds[i]))
    return spds

def spd_aggregator(data_to_label, voters=[], group=[], fairness=fairness_metrics):
    # predictions from teachers
    if voters == []:
        voters = teachers.copy()
    if fairness == []:
        fairness = fairness_metrics.copy()
    preds = []
    for tchr in voters:
        pred = tchr.model.predict(data_to_label)
        pred = np.round(pred)
        preds.append([p[0] for p in pred])
    preds = np.asarray(preds, dtype=np.int32)

    weighs = get_spd_weighs(data_to_label, preds, group, fairness=fairness)
    labels = []
    for x in range(np.shape(preds)[1]):
        pred = list(preds[:,x])
        for i in range(len(pred)):
            if fairness[i] < 0.1:
                pred = pred + [pred[i]]*weighs[i]
        n_y_x = np.bincount(pred)
        labels.append(np.argmax(n_y_x))
        
    return np.asarray(labels), weighs

def methode_2(data_to_label, voters=[], group=[], fairness=fairness_metrics):
    # predictions from teachers
    if voters == []:
        voters = teachers.copy()
    if fairness == []:
        fairness = fairness_metrics.copy()
    preds = []
    for tchr in voters:
        pred = tchr.model.predict(data_to_label)
        pred = np.round(pred)
        preds.append([p[0] for p in pred])
    preds = np.asarray(preds, dtype=np.int32)

    weighs = get_spd_weighs_2(data_to_label, preds, group, fairness=fairness)
    labels = []
    for x in range(np.shape(preds)[1]):
        pred = list(preds[:,x])
        for i in range(len(pred)):
            if fairness[i] < 0.1:
                pred = pred + [pred[i]]*weighs[i]
        n_y_x = np.bincount(pred)
        labels.append(np.argmax(n_y_x))
        
    return np.asarray(labels), weighs

def update_aggregator(current):
    choice = input("1. Plurality \t 2. LNMax\t 3. GNMax \n4. Only Fair \t 5. Only unfair\n (0 to exit)>>> ")
    if choice == "1":
        return plurality
    elif choice == "2":
        return laplacian_noisy_vote
    elif choice == "3":
        return gaussian_noisy_vote
    elif choice == "4":
        """ global methode
        choice = input("Methode : 1. Plurality \t 2. LNMax\t 3. GNMax\n>>> ")
        if choice == "3":
            methode = gaussian_noisy_vote
        elif choice == "2":
            methode = laplacian_noisy_vote
        else:
            methode = plurality """
           
        print(f"Only Fair teachers participates using {methode.__name__}")
        return only_fair
    elif choice == "5":
        """ global methode
        choice = input("Methode : 1. Plurality \t 2. LNMax\t 3. GNMax\n>>> ")
        if choice == "3":
            methode = gaussian_noisy_vote
        elif choice == "2":
            methode = laplacian_noisy_vote
        else:
            methode = plurality """
           
        print(f"Only unfair teachers participates using {methode.__name__}") 
        return only_unfair
    else:
        return current
    
