import sys
from teachers import *
from student import *
from aggregator import *
from data_loader import load_student_data
#import matplotlib as mpl
import matplotlib.pyplot as plt
from analysis import *
import warnings
from teacher_ensemble import Ensemble, Teacher


colors = plt.rcParams["axes.prop_cycle"].by_key()['color']
color_index = 0

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

def get_agg(cf):
    if cf == "All":
        aggregator = plurality
    elif cf == "Only fair":
        aggregator = only_fair
    elif cf == "Only unfair":
        aggregator = only_unfair
    elif cf == "Weighed vote":
        aggregator = weighed_vote
    elif cf == "Fairfed":
        aggregator = fair_fed_agg
    elif cf == "Methode1":
        aggregator = spd_aggregator
    elif cf == "Methode2":
        aggregator = methode_2
    
    return aggregator
dataset = "acsemployment_bis"
# student data
(x_train, x_test, y_train, y_test, s_train, s_test) = load_student_data("AK")

conf = ["All", "Only fair", "Only unfair", "Weighed vote", "Fairfed", "Methode1", "Methode2"]
for nb_teachers in [5, 10, 30, 50]:
    fig, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2,3, sharey=True)
    st_fairness = {}
    print(">>> ", nb_teachers, " teachers")
    for cf in conf:
        st_fairness[cf] = []
    for nb_fair_tchrs in range(1, nb_teachers):
        tchrs_ensemble = Ensemble(nb_teachers, nb_fair_tchrs)
        update_teachers(tchrs_ensemble.tchrs)

        eod = []
        for tchrs in tchrs_ensemble.tchrs:
            eod.append(tchrs.metrics["EOD"])
        set_metrics(eod)

        for cf in conf:
            print(f'>>> case : {cf}')
            aggregator = get_agg(cf)
            y_train, _ = aggregator(x_train, group=s_train)
            yhat_test, _ = aggregator(x_test, group=s_test)
            st_model = train_student(x_train, y_train, verbose=False)
            st_stats = fairness(st_model, x_test, yhat_test, s_test)
            st_fairness[cf].append(st_stats["EOD"])
    color_index = 1
    for cf in conf:
        if cf == "All":
           continue
        elif cf == "Only fair":
           ax1.plot(list(range(1, nb_teachers)), st_fairness[cf], color=colors[color_index], label=cf)
           ax1.plot(list(range(1, nb_teachers)), st_fairness["All"], color=colors[0])
        elif cf == "Only unfair":
           ax2.plot(list(range(1, nb_teachers)), st_fairness[cf], color=colors[color_index], label=cf)
           ax2.plot(list(range(1, nb_teachers)), st_fairness["All"], color=colors[0])
        elif cf == "Weighed vote":
           ax3.plot(list(range(1, nb_teachers)), st_fairness[cf], color=colors[color_index], label=cf)
           ax3.plot(list(range(1, nb_teachers)), st_fairness["All"], color=colors[0])
        elif cf == "Fairfed":
           ax4.plot(list(range(1, nb_teachers)), st_fairness[cf], color=colors[color_index], label=cf)
           ax4.plot(list(range(1, nb_teachers)), st_fairness["All"], color=colors[0])
        elif cf == "Methode1":
           ax5.plot(list(range(1, nb_teachers)), st_fairness[cf], color=colors[color_index], label=cf)
           ax5.plot(list(range(1, nb_teachers)), st_fairness["All"], color=colors[0])
        elif cf == "Methode2":
           ax6.plot(list(range(1, nb_teachers)), st_fairness[cf], color=colors[color_index], label=cf)
           ax6.plot(list(range(1, nb_teachers)), st_fairness["All"], color=colors[0], label="All")

        color_index += 1
    fig.legend(loc="outside upper left",ncol=4)
    ax1.set_ylabel("Student fairness")
    ax5.set_xlabel("Number of fair teachers")
    plt.savefig("../img/archive_"+ str(nb_teachers) + "/st_fairness_variations_" + str(nb_teachers) + "_teachers.png")
    


