import numpy as np

def fairness(model, x_test, y_test, group_test):
    yhat = np.round(model.predict(x_test))
    acc = float(format(np.mean(yhat[(y_test == 1)]), "0.4f"))

    p_grp_tpr = np.mean(yhat[(y_test == 1) & (group_test == 1)])
    up_grp_tpr = np.mean(yhat[(y_test == 1) & (group_test == 2)])
    # equality of difference (opportinuty)
    eod = float(format(p_grp_tpr - up_grp_tpr, ".4f"))

    # statistical parity difference
    p_grp = np.mean(yhat[(group_test == 1)])
    up_grp = np.mean(yhat[(group_test == 2)])
    spd = float(format(p_grp - up_grp, ".4f"))

    return {"EOD": eod, "SPD": spd, "ACC": acc}

def stats(teachers, x_test, y_test, s_test):
    accuracies = []
    eod = []
    spd = []
    for teacher in teachers:
        stat = fairness(teacher, x_test, y_test, s_test)
        accuracies.append(stat["ACC"])
        eod.append(stat["EOD"])
        spd.append(stat["SPD"])
    return accuracies, eod, spd


