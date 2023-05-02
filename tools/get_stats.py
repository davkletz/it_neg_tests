import math

from joblib import dump, load
import numpy as np

def verb_diversity(path):

    detail_verbs = load(f"{path}")

    tot_used_verbs = 0

    '''for verb in detail_verbs:
        if detail_verbs[verb] !=0:
            tot_used_verbs += 1
    '''


    score = list(detail_verbs.values())

    max_val = np.max(score)

    for i in range(len(score)):
        score[i] = score[i]/max_val


    return np.std(score)


name_m = "camembert-large"
name_m = "flaubert_large_cased"
name_m = "bert-base-multilingual-cased"
name_m = "camembert-base"
name_m = "flaubert_base_cased"

a = f"/home/dkletz/tmp/pycharm_project_99/2022-23/neg-eval-set/evaluation_script/Inputs/{name_m}/detail_verbs_{name_m}.joblib"
b = verb_diversity(a)

#print(b)



list_nb_verbs = []
list_good = []
list_per = []
list_diversity = []

for name_model in ["dbmdz/bert-base-italian-cased", "bert-base-multilingual-cased",
                   "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"]:

    model_name = name_model
    if "/" in name_model:
        model_name = name_model.split("/")[-1]
        # current_model_name = current_model_name[0] + "_" + current_model_name[1]



    path = f"/home/dkletz/tmp/pycharm_project_99/stages/viola/Inputs"

    a = load(f"{path}/{model_name}/total_sentences_mono_{model_name}.joblib")
    list_nb_verbs.append(a)

    b = load(f"{path}/{model_name}/tot_good_preds_mono_{model_name}.joblib")
    list_good.append(b)

    c = (round((b/a)*10000))/100
    list_per.append(c)

    d = verb_diversity(f"{path}/{model_name}/detail_verbs_mono_{model_name}.joblib")
    print("\n")
    print(d)
    print(math.isnan(d))
    if not math.isnan(d):
        list_diversity.append(round(d*10000)/100)
    else:
        list_diversity.append(0)

print("DAA")

vall = ""
for elem in list_nb_verbs:
    vall += str(elem) + " & "

print(vall[:-2])


vall = ""
for elem in list_good:
    vall += str(elem) + " & "

print(vall[:-2])



vall = ""
for elem in list_per:
    vall += str(elem) + " & "

print(vall[:-2])


vall = ""
for elem in list_diversity :
    vall += str(elem) + " & "

print(vall[:-2])





