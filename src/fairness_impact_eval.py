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

if len(sys.argv) < 3:
    print("Usage : fairness_impact_eval.py <nb_teachers> <nb_fair_wished>")
    exit(1)

dataset = "acsemployment_bis"
nb_teachers = int(sys.argv[1])
nb_fair_tchrs = int(sys.argv[2]) # wished

# creating teachers
tchrs_ensemble = Ensemble(nb_teachers, nb_fair_tchrs) 
# prepare datasets !
#subsets, S = get(dataset, nb_teachers, nb_fair_tchrs=nb_fair_tchrs)

# train teachers
#tchrs_ensemble.parallel_training()
update_teachers(tchrs_ensemble.tchrs)

accuracies, eod, spd, rec = [], [], [], []
for tchrs in tchrs_ensemble.tchrs:
    accuracies.append(tchrs.metrics["ACC"])
    eod.append(tchrs.metrics["EOD"])
    spd.append(tchrs.metrics["SPD"])
    rec.append(tchrs.metrics["REC"])
set_metrics(eod)

name = dataset + "_" + str(nb_fair_tchrs) + "_fair"+ ".png"

# load student dataset
(x_train, x_test, y_train, y_test, s_train, s_test) = load_student_data("AK")

""" labels, spd_ws = spd_aggregator(x_train, group=s_train)
print("nb of 0 in labels = ", np.count_nonzero(labels == 0), end="\n#################################\n")
print(spd_ws, end="\n#########\n")
labels, _ = fair_fed_agg(x_train)
print("nb of 0 in labels = ", np.count_nonzero(labels == 0), end="\n#################################\n")
print(spd_ws)
exit(1)
 """
confs = ["All", "Only fair", "Only unfair"]

fig, (ax1, ax2, ax3)= plt.subplots(1, 3, sharey=True)
b_width = 0.3
x = range(len(accuracies))
# teachers hist 
ax1.bar(x, eod, width = b_width, color=["green" for _ in eod], hatch="//")
cp_state = states.copy()
cp_state.pop(2)
ax1.set_xticks(x, ['' for _ in range(nb_teachers)])
ax1.set_yticks(np.arange(0, 1.1, step=0.1))
ax1.set_ylim([0,1.1])
ax1.set_xlabel("Teachers")
ax1.set_ylabel("Metrics")

xticks = ["all", "f", "uf", "wv"]
stats = {}
for cf in confs:
    # setting conf
    if cf != "All" and (nb_fair_tchrs == 0 or nb_fair_tchrs == nb_teachers):
        break
    print(f'{cf} teachers')
    if cf == "All":
        aggregator = plurality
        x = 1
    elif cf == "Only fair":
        aggregator = only_fair
        x += 3*b_width/2
    else:
        if nb_fair_tchrs == nb_teachers:
            continue
        aggregator = only_unfair
        x += 3*b_width/2
    y_train, _ = aggregator(x_train)
    yhat_test, consensus = aggregator(x_test)
    st_model = train_student(x_train, y_train, verbose=False)
    ev1, ev2 = eval_student_model(st_model, x_test, y_test, yhat_test, verbose=False)
    st_stats = fairness(st_model, x_test, yhat_test, s_test)
    stats[cf] = [ev1[1], ev2[1], st_stats["EOD"]]
    ax2.bar([x], 4*consensus, width = b_width, color="red")
    color_index += 1

x=1
color_index = 0
for cf, stat in stats.items():
    ax3.bar([x], stat, width = b_width, color=["#fcba03", "#8c6908", "green"], bottom=[0,0,0], hatch=["", "", "//"])
    x += 3*b_width/2

ax3.set_xlabel("Student")

# ###############
# Give 1 more voice to each fair teacher
# ###############
print(f'Simple weighed vote')
aggregator = weighed_vote
y_train, _= aggregator(x_train)
yhat_test, consensus = aggregator(x_test)
st_model = train_student(x_train, y_train, verbose=False)
st_stats = fairness(st_model, x_test, yhat_test, s_test)
ev1, ev2 = eval_student_model(st_model, x_test, y_test, yhat_test, verbose=False)
stat = [ev1[1], ev2[1], st_stats["EOD"]]
ax3.bar([x], stat, width = b_width, color=["#fcba03", "#8c6908", "green"],  label=["ACC-Labeled data", "ACC - True labels", "EOD"], hatch=["", "", "/"])
ax2.bar([x], 4*consensus, width = b_width, color="red", label="consensus")
x += 3*b_width/2

# ###############
# weighs computes using fairfed
# ###############

print(f"fairfed's weighed vote")
aggregator = fair_fed_agg
y_train, _= aggregator(x_train)
yhat_test, _ = aggregator(x_test)
st_model = train_student(x_train, y_train, verbose=False)
st_stats = fairness(st_model, x_test, yhat_test, s_test)
ev1, ev2 = eval_student_model(st_model, x_test, y_test, yhat_test, verbose=False)
stat = [ev1[1], ev2[1], st_stats["EOD"]]
ax3.bar([x], stat, width = b_width, color=["#fcba03", "#8c6908", "green"], hatch=["", "", "/"])
x += 3*b_width/2

# ###############
# weighs computes using spds
# ###############

print(f"Spd's weighed vote")
aggregator = spd_aggregator
y_train, _= aggregator(x_train, group=s_train)
yhat_test, _ = aggregator(x_test, group=s_test)
st_model = train_student(x_train, y_train, verbose=False)
st_stats = fairness(st_model, x_test, yhat_test, s_test)
ev1, ev2 = eval_student_model(st_model, x_test, y_test, yhat_test, verbose=False)
stat = [ev1[1], ev2[1], st_stats["EOD"]]
ax3.bar([x], stat, width = b_width, color=["#fcba03", "#8c6908", "green"], hatch=["", "", "/"])



ax3_xtick = xticks + ["ff", "spd"]
ax3.set_xticks([1 + i*3*b_width/2 for i in range(len(ax3_xtick))], ax3_xtick) 
ax2.set_xticks([1 + i*3*b_width/2 for i in range(len(xticks))], xticks) 

ax2.set_xlabel("Teachers's concensus")
fig.legend(loc="outside upper left",ncol=4)
path = "../img/archive_" + str(nb_teachers) + "/" + name

while os.path.isfile(path):
    path = path[:-4] + "_.png" 
plt.savefig(path)


