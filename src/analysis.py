import numpy as np

def fairness(model, x_test, y_test, group_test):
    yhat = np.round(model.predict(x_test))
    acc = float(format(model.evaluate(x_test, y_test)[1], "0.4f"))
    #acc = float(format(np.mean(yhat[(y_test == 1)]), "0.4f"))

    p_grp_tpr = np.mean(yhat[(y_test == 1) & (group_test == 1)])
    up_grp_tpr = np.mean(yhat[(y_test == 1) & (group_test == 2)])
    # equality of difference (opportinuty)
    eod = float(format(p_grp_tpr - up_grp_tpr, ".4f"))

    # statistical parity difference
    p_grp = np.mean(yhat[(group_test == 1)])
    up_grp = np.mean(yhat[(group_test == 2)])
    spd = float(format(p_grp - up_grp, ".4f"))

    return {"EOD": eod, "SPD": spd, "ACC": acc}

def stats(nb_teachers, teachers, subsets):
    accuracies = []
    eod = []
    spd = []
    for i in range(nb_teachers):
        params = [subsets[i][1], subsets[i][3], subsets[i][5]]
        stat = fairness(teachers[i], *params)
        accuracies.append(stat["ACC"])
        eod.append(stat["EOD"])
        spd.append(stat["SPD"])
    return accuracies, eod, spd


