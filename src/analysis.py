import numpy as np
import math

def mean(myarray):
    mn = np.mean(myarray)
    return 0 if math.isnan(mn) else mn

def fairness(model, x_test, y_test, group_test):
    yhat = np.round(model.predict(x_test))
    ev = model.evaluate(x_test, y_test)
    acc = float(format(ev[1], "0.4f"))
    rec = float(format(ev[2], ".4f"))

    p_grp_tpr = mean(yhat[(y_test == 1) & (group_test == 1)])
    up_grp_tpr = mean(yhat[(y_test == 1) & (group_test == 2)])
    
    # equality of difference (opportinuty)
    eod = float(format(abs(p_grp_tpr - up_grp_tpr), ".4f"))

    # statistical parity difference
    p_grp = mean(yhat[(group_test == 1)])
    up_grp = mean(yhat[(group_test == 2)])
    spd = float(format(abs(p_grp - up_grp), ".4f"))

    return {"EOD": eod, "SPD": spd, "ACC": acc, "REC": rec}

def stats(nb_teachers, teachers, subsets, S):
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


