import numpy as np

teachers = []
def init_teachers(tchrs):
    global teachers
    teachers = tchrs

def agg_noisy_vote(data_to_label, gamma=0.1):
    # predictions from teachers
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
        # adding noize to votes
        for i in range(len(n_y_x)):
            n_y_x[i] += np.random.laplace(loc=0.0, scale=gamma)
        labels.append(np.argmax(n_y_x))
    return labels

# other aggragation methods here
