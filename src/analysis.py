import numpy as np
import math

def fairness(model, x_test, y_test, group_test):
    yhat = np.round(model.predict(x_test))
    ev = model.evaluate(x_test, y_test)
    acc = float(format(ev[1], "0.4f"))
    rec = float(format(ev[2], ".4f"))

    p_grp_tpr = np.mean(yhat[(y_test == 1) & (group_test == 1)])
    up_grp_tpr = np.mean(yhat[(y_test == 1) & (group_test == 2)])
    # equality of difference (opportinuty)
    if math.isnan(p_grp_tpr):
        p_grp_tpr = 0
    if math.isnan(up_grp_tpr):
        up_grp_tpr = 0
    eod = float(format(abs(p_grp_tpr - up_grp_tpr), ".4f"))

    # statistical parity difference
    p_grp = np.mean(yhat[(group_test == 1)])
    up_grp = np.mean(yhat[(group_test == 2)])
    if math.isnan(p_grp):
        p_grp = 0
    if math.isnan(up_grp):
        up_grp = 0
    spd = float(format(abs(p_grp - up_grp), ".4f"))

    return {"EOD": eod, "SPD": spd, "ACC": acc, "REC": rec}

def stats(nb_teachers, teachers, subsets):
    accuracies = []
    eod = []
    spd = []
    rec = []
    for i in range(nb_teachers):
        params = [subsets[i][1], subsets[i][3], subsets[i][5]]
        stat = fairness(teachers[i], *params)
        accuracies.append(stat["ACC"])
        eod.append(stat["EOD"])
        spd.append(stat["SPD"])
        rec.append(stat["REC"])
    return accuracies, eod, spd, rec


