import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

import torch
from joblib import load, dump
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

from sklearn.neural_network import MLPClassifier

import numpy as np
from sklearn.preprocessing import StandardScaler
import re
from random import shuffle, seed
import nltk

seed(42)

# from sklearn.model_selection import train_test_split


from mlconjug3 import Conjugator



######



nb_sents_to_catch = 10000






########################
### useful functions ###
########################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





conjugator = Conjugator(language="it") # così usa l'italiano

def check_conjugation(verb, conjugation):
    """
    Verifie si une forme est bien une flexion d'un verbe.

    Args:
        verb (str): The verb to check.
        conjugation (str): Forme a verifier.

    Returns:
        bool: Vrai si la forme est bien une flexion du verbe.

    """
    try:
        verb_conjs = conjugator.conjugate(verb).iterate()

    except:
        for k in range(10000):
            l = 0

        print(f'except : {verb}, {conjugation}')

        return False


    if  (('Indicativo', 'Indicativo presente', 'egli/ella', conjugation) in verb_conjs) or (('Indicativo', 'Indicativo presente', '3s', conjugation) in verb_conjs):
        return True

    return False



def build_masked_context_negative(name_available, profession_available, verb):

    context_available = "NOM è MET che non ha l'abitudine di ACTION."# it

    new_context_sentence_available = context_available.replace('NOM', name_available)
    new_context_sentence_available = new_context_sentence_available.replace('MET', profession_available)
    new_context_sentence_available = new_context_sentence_available.replace('ACTION', verb)

    return new_context_sentence_available


def build_masked_context_positive(name_available, profession_available, verb):

    context_available = "NOM è MET che ha l'abitudine di ACTION."# it

    new_context_sentence_available = context_available.replace('NOM', name_available)
    new_context_sentence_available = new_context_sentence_available.replace('MET', profession_available)
    new_context_sentence_available = new_context_sentence_available.replace('ACTION', verb)

    return new_context_sentence_available

def build_masked_target_positive(current_pronouns_maj, mask_token):

    context_available = "PRON_maj MASK molto spesso."# it
    new_context_sentence_available = context_available.replace('PRON_maj', current_pronouns_maj)
    new_context_sentence_available = new_context_sentence_available.replace('MASK', mask_token)

    return new_context_sentence_available



def build_masked_target_negative(current_pronouns_maj, mask_token):

    context_available = "PRON_maj non MASK molto spesso."# it
    new_context_sentence_available = context_available.replace('PRON_maj', current_pronouns_maj)
    new_context_sentence_available = new_context_sentence_available.replace('MASK', mask_token)

    return new_context_sentence_available



def build_array(sourcefile):
    """"
    The function builds an array out of an inputfile,
    one array entry per line
    """
    array_in_construction=[]
    for line in sourcefile:
        #print(line)
        cleanline=line.strip('\n ')
        array_in_construction.append(cleanline)
    return array_in_construction



########################
### useful functions ###
########################


def make_and_encode_batch(current_batch, tokenizer, model, device, batch_verbs, name_available, profession_available,
                          current_pronouns_maj, found):

    current_found = found  # true if the current batch contained good sentences
    good_pred = 0
    detail_verbs = []

    # get the predicted tokens for the batch of sentences
    predictions = encode_batch(current_batch, tokenizer, model, device)
    new_sentence = None

    # for each prediction, check if the model predicted the same verb that was in the context sentence
    for i, prediction_available in enumerate(predictions):
        good_verb = batch_verbs[i]  # the desired verb

        if check_conjugation(good_verb, prediction_available):
            # outputs True value if the prediction is the 3rd person plural of the desired verb
            detail_verbs.append(good_verb)
            good_pred += 1
            good_dico = {"name_available": name_available, "profession_available": profession_available,
                         "verb": good_verb, "current_pronouns_maj": current_pronouns_maj,
                         "masked_prediction": prediction_available}

            if not current_found:
                # once a good sentence is found, the "found" value is set to true
                # and the "new_sentence" value is the dictionary of all elements in the sentence
                new_sentence = good_dico

                current_found = True
                # if not complete_check: ########
                #    break
    return new_sentence, current_found, good_pred, detail_verbs


def encode_batch(current_batch, tokenizer, model, device):
    with torch.no_grad():
        # encode sentences
        encoded_sentence = tokenizer.batch_encode_plus(current_batch, padding=True, return_tensors="pt").to(device)
        # get the mask-token index in the sentence
        mask_tokens_index = torch.where(encoded_sentence['input_ids'] == tokenizer.mask_token_id)
        # print(mask_tokens_index)
        # get logits vectors
        tokens_logits = model(**encoded_sentence)
        # print(tokens_logits)
        # print(tokens_logits['logits'].shape)

        # get the mask-token logit
        mask_tokens_logits = tokens_logits['logits'][mask_tokens_index]
        # print(mask_tokens_logits.shape)

        # get the k highest logits
        top_tokens = torch.topk(mask_tokens_logits, 1, dim=1).indices  # .tolist()
        # print(top_tokens)

        # decode the batch of tokens, i.e. get the predicted tokens (each token is represented by an index in the vocabulary)
        predicted_tokens = tokenizer.batch_decode(top_tokens)
        # print(predicted_tokens)

    return predicted_tokens


##############################
### paisa "non" extraction ###
##############################


size_test = 10000

# select the italian model to test
model = AutoModel.from_pretrained('dbmdz/bert-base-italian-cased')
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-italian-cased')






with open(r"../data/paisa.raw.utf8", encoding='utf8') as infile:
  paisa = infile.read()



wiki_pattern = r"<text.*wiki.*(?:\n.*)+?\n</text>\n" # finds all texts containing "wiki" in their tag's url

paisa_wiki = re.findall(wiki_pattern, paisa)

print(f"Number of texts from a site containing 'wiki' in their URL: {len(paisa_wiki)}")


i = 0
print(len(paisa_wiki))
shuffle(paisa_wiki)


paisa_wiki = paisa_wiki[:1000]



Cn_patt = r"\b[Nn]on\b.*\..*\."  # finds the negation in a sentence
Tn_patt = r"\..*\b[Nn]on\b.*\."  # finds the negation in a sentence


Cn = []
CpTp = []
CpTn = []


CnTp = []
CnTn = []

shuffle(paisa_wiki)


for text in paisa_wiki:

    if len(CnTn) >= nb_sents_to_catch and len(CnTp) >= nb_sents_to_catch and len(CpTn) >= nb_sents_to_catch and len(CpTp) >= nb_sents_to_catch:
        break


    if i%100 == 0:
        print(f"Text number {i}, {len(CpTn)} CpTn found")
    i += 1
    tokens = nltk.sent_tokenize(text)

    for element in range(len(tokens)-1):
        new_sent = tokens[element] + " " + tokens[element+1]


        if len(new_sent) > 25 and not "\n" in new_sent:


            matches1 = re.search(Cn_patt, new_sent)
            if matches1:
                double_neg = re.search(Tn_patt, new_sent)
                if double_neg:
                    CnTn.append(new_sent)
                else:
                    CnTp.append(new_sent)
            else:
                matches2 = re.search(Tn_patt, new_sent)
                if matches2:
                    CpTn.append(new_sent)
                else:
                    CpTp.append(new_sent)




sent_neg = CnTn + CnTp + CpTn
sent_pos = CpTp

shuffle(sent_neg)
shuffle(sent_pos)

size_test = min(size_test, len(sent_neg), len(sent_pos))

# select a fixed numb of sentences to test
sent_neg = sent_neg[:size_test]
sent_pos = sent_pos[:size_test]

size_batch = 4

### extract CLS
# for each set of sentences, we encode each sentence
cls_encodings_neg = np.zeros((size_test, 768))
cls_encodings_pos = np.zeros((size_test, 768))

for sent_list in [sent_neg, sent_pos]:

    #print(len(sent_list))
    nb_batch = len(sent_list) // size_batch
    for k in range(nb_batch):
        current_batch = sent_list[k * size_batch:(k + 1) * size_batch]
        batch_encoded = tokenizer.batch_encode_plus(current_batch, padding=True, add_special_tokens=True, return_tensors="pt")
        batch_encoded = batch_encoded.to(device)
        #print(len(batch_encoded))

        # then extract only the outputs for each sentence
        with torch.no_grad():
            tokens_outputs = model(**batch_encoded)



        # for each set of outputs we only keep the one of the CLS token, namely the first token of each sentence
        cls_encodings = tokens_outputs.last_hidden_state[:, 0, :]

        cls_encodings = cls_encodings.cpu().numpy()

        if sent_list == sent_neg:
            cls_encodings_neg[k * size_batch:(k + 1) * size_batch] = cls_encodings
        elif sent_list == sent_pos:
            cls_encodings_pos[k * size_batch:(k + 1) * size_batch] = cls_encodings


    if len(sent_list) % size_batch != 0:
        current_batch = sent_list[nb_batch * size_batch:]
        batch_encoded = tokenizer.batch_encode_plus(current_batch, padding=True, add_special_tokens=True, return_tensors="pt").to(device)
        #print(len(batch_encoded))

        # then extract only the outputs for each sentence
        with torch.no_grad():
            tokens_outputs = model(**batch_encoded)

        # for each set of outputs we only keep the one of the CLS token, namely the first token of each sentence
        cls_encodings = tokens_outputs.last_hidden_state[:, 0, :]

        cls_encodings = cls_encodings.cpu().numpy()

        if sent_list == sent_neg:
            cls_encodings_neg[nb_batch * size_batch:] = cls_encodings
        elif sent_list == sent_pos:
            cls_encodings_pos[nb_batch * size_batch:] = cls_encodings


# we use 90% of data as training and 10% as test
train_size = round(size_test * 0.9)
train = np.concatenate((cls_encodings_pos[:train_size], cls_encodings_neg[:train_size]), 0)
labels = np.concatenate((np.zeros(train_size), np.ones(train_size)))
test = np.concatenate((cls_encodings_pos[train_size:], cls_encodings_neg[train_size:]), 0)
test_size = int(size_test - train_size)
test_lab = np.concatenate((np.zeros(test_size), np.ones(test_size)))

# data normalization
scaler = StandardScaler()
scaler.fit(train)
dati_scaled = scaler.transform(train)

X = dati_scaled
test = scaler.transform(test)
# print(test)
# print(test_lab)

y = labels





###########################
###########################
###########################
###########################
###########################
### masked template set ###
###########################

###########################
###########################
###########################
###########################
###########################

model_mask = AutoModelForMaskedLM.from_pretrained('dbmdz/bert-base-italian-cased').to(device)

# load names, professions and verbs for the templates
path = r"../Inputs"
fName_file_path = f"{path}/100_names_f.txt"
mName_file_path = f"{path}/100_names_m.txt"
fProf_file_path = f"{path}/100_mestieri_f.txt"
mProf_file_path = f"{path}/100_mestieri_m.txt"
hypo_file_path = f"{path}/frasi_it.txt"

fName_file = open(fName_file_path, "r")
mName_file = open(mName_file_path, "r")
fProf_file = open(fProf_file_path, "r")
mProf_file = open(mProf_file_path, "r")

list_verbs = load(f"{path}/base_verbs.joblib")

# dictionaries of names, professions and pronouns indexed by gender for template construction
professionsarray = {"f": build_array(fProf_file),
                    "m": build_array(mProf_file)}  # buildarray is a function for creating lists from txt files
fnamearray = build_array(fName_file)
mnamearray = build_array(mName_file)
name_arrays = {"f": fnamearray, "m": mnamearray}
pronouns_maj = {"f": "Lei", "m": "Lui"}

# set up list for patterns that, for the CpTp setting, predict for the mask the same verb that was in the context
list_good_patterns_model = []

total_sentences = 0  # counts tried sentences
tot_good_preds = 0  # counts sentences with repetition
detail_verbs = {v: 0 for v in
                list_verbs}  # counts, for each verb, how many times it is repeated in the mask if present in context

size_batches = 100

nb_found_sentences = 0


nb_max_patterns = 100


for gender in ["f", "m"]:
    current_pronouns_maj = pronouns_maj[gender]

    for name_available in name_arrays[gender]:
        batch_CpTp = []  # batch of CpTp to try in this cycle

        batch_verbs = []  # batch of verbs to try in this cycle


        for profession_available in professionsarray[gender]:

            if len(list_good_patterns_model)>=nb_max_patterns:
                break

            current_list_verbs = list_verbs.copy()
            shuffle(current_list_verbs)

            found = False  # to stop when a good verb is found

            for verb_available in current_list_verbs:
                # print(f"current verb : {verb_available}")
                # if not complete_check and found:
                #    break


                current_Cp = build_masked_context_positive(name_available, profession_available, verb_available)
                current_Tp = build_masked_target_positive(current_pronouns_maj, mask_token=tokenizer.mask_token)

                batch_CpTp.append(current_Cp +" " + current_Tp)


                batch_verbs.append(verb_available)
                total_sentences += 1

                if total_sentences % 1000 == 0:
                    print(f"current : {total_sentences}, {len(list_good_patterns_model)}")

                # get the result at the end of the batch
                if len(batch_CpTp) == size_batches:
                    #print(len(batch_CpTp))
                    #print(len(batch_verbs))
                    new_sentence, found, nb_good_pred, found_verbs = make_and_encode_batch(batch_CpTp, tokenizer,
                                                                                           model_mask, device,
                                                                                           batch_verbs, name_available,
                                                                                           profession_available,
                                                                                           current_pronouns_maj, found)


                    tot_good_preds += nb_good_pred
                    if new_sentence != None:
                        list_good_patterns_model.append(new_sentence)
                    batch_CpTp = []
                    batch_verbs = []
                    for found_verb in found_verbs:
                        nb_found_sentences += 1
                        detail_verbs[found_verb] += 1  # add one repetition to the count for the found verb

            # repetition for what is left out of the last batch
            if len(batch_CpTp) > 0:
                new_sentence, found, nb_good_pred, found_verbs = make_and_encode_batch(batch_CpTp, tokenizer,
                                                                                       model_mask, device, batch_verbs,
                                                                                       name_available,
                                                                                       profession_available,
                                                                                       current_pronouns_maj, found)

                tot_good_preds += nb_good_pred
                if new_sentence != None:
                    list_good_patterns_model.append(new_sentence)

                batch_verbs = []
                batch_CpTp = []
                for found_verb in found_verbs:
                    detail_verbs[found_verb] += 1


# create the CpTp set
template_sentences_CpTp = []
template_sentences_CpTn = []
template_sentences_CnTp = []
template_sentences_CnTn = []



for pattern in list_good_patterns_model:
    # build sentences putting the conjugated verb instead of the mask
    Cp = build_masked_context_positive(pattern["name_available"], pattern["profession_available"],
                                       pattern["verb"])
    Cn = build_masked_context_negative(pattern["name_available"], pattern["profession_available"],
                                   pattern["verb"])
    Tp = build_masked_target_positive(pattern["current_pronouns_maj"], tokenizer.mask_token)
    Tn = build_masked_target_negative(pattern["current_pronouns_maj"], tokenizer.mask_token)

    template_sentences_CpTp.append(Cp + " " + Tp)
    template_sentences_CpTn.append(Cp + " " + Tn)
    template_sentences_CnTp.append(Cn + " " + Tp)
    template_sentences_CnTn.append(Cn + " " + Tn)




# extract CLS for each template sentence
# for each set of sentences, we encode each sentence

cls_temp_CnTn = np.zeros((len(template_sentences_CnTn), 768))
cls_temp_CpTp = np.zeros((len(template_sentences_CpTp), 768))
cls_temp_CpTn = np.zeros((len(template_sentences_CpTn), 768))
cls_temp_CnTp = np.zeros((len(template_sentences_CnTp), 768))

template_sentences = {"CpTp" : template_sentences_CpTp, "CpTn" : template_sentences_CpTn, "CnTp" : template_sentences_CnTp, "CnTn" : template_sentences_CnTn }

cls_temps = {}

for construction in template_sentences:
    sent_list = template_sentences[construction]
    current_cls_temps = np.zeros((len(template_sentences_CpTn), 768))
    nb_batch = len(sent_list) // size_batch
    print(nb_batch)
    for k in range(nb_batch):
        #print(f"currnet k : {k}")
        current_batch = sent_list[k * size_batch:(k + 1) * size_batch]
        batch_encoded = tokenizer.batch_encode_plus(current_batch, padding=True, add_special_tokens=True,
                                                    return_tensors="pt").to(device)

        #batch_encoded = tokenizer.batch_encode_plus(sent_list, padding=True, add_special_tokens=True, return_tensors="pt").to(device)

        # then extract only the outputs for each sentence
        with torch.no_grad():
            tokens_outputs = model(**batch_encoded)

        # for each set of outputs we only keep the one of the CLS token, namely the first token of each sentence
        cls_encodings = tokens_outputs.last_hidden_state[:, 0, :]

        cls_encodings = cls_encodings.cpu().numpy()

        current_cls_temps[k * size_batch:(k + 1) * size_batch] = cls_encodings



    r_eq = len(sent_list) % size_batch
    if  r_eq !=0:
        current_batch = sent_list[nb_batch * size_batch:]
        batch_encoded = tokenizer.batch_encode_plus(current_batch, padding=True, add_special_tokens=True,
                                                    return_tensors="pt").to(device)

        # batch_encoded = tokenizer.batch_encode_plus(sent_list, padding=True, add_special_tokens=True, return_tensors="pt").to(device)

        # then extract only the outputs for each sentence
        with torch.no_grad():
            tokens_outputs = model(**batch_encoded)

        # for each set of outputs we only keep the one of the CLS token, namely the first token of each sentence
        cls_encodings = tokens_outputs.last_hidden_state[:, 0, :]

        cls_encodings = cls_encodings.cpu().numpy()

        current_cls_temps[nb_batch * size_batch:] = cls_encodings

    cls_temps[construction] = current_cls_temps



print(f"details : {cls_temps.keys()}")

for construction in cls_temps:

    np.random.shuffle(cls_temps[construction])


print(f"SIZE : {len(template_sentences['CpTp'])}")

current_size = len(template_sentences['CpTp'])
for construction in cls_temps:
    current_size = min(current_size, len(cls_temps[construction]))


for construction in cls_temps:
    cls_temps[construction] = cls_temps[construction][:current_size]




print(f"details : {cls_temps.keys()}")


############################
### masked template test ###
############################


test_temp = np.concatenate((cls_temps["CpTp"], cls_temps["CpTn"], cls_temps["CnTp"], cls_temps["CnTn"]))
test_temp_lab = np.concatenate((np.zeros(current_size), np.ones(current_size),  np.ones(current_size),  np.ones(current_size)))


scaler.fit(test_temp)
#train = scaler.transform(train_temp)
test_2 = scaler.transform(test_temp)






########################################
### classifier creation and training ###
########################################


paisa_result = []
template_result = []

# set up the MLP classifier
# solver : adam or sgd
# hidden_layer_sizes : 40,40 or 350,350
# alpha : between 1e-5 and 1e-2
for hl in [(40, 40), (350, 350)]:
    for a in [1e-2, 1e-3, 1e-4, 1e-5]:
        for solv in ["adam", "sgd"]:
            clf = MLPClassifier(solver="adam", alpha=a,
                                hidden_layer_sizes=hl, random_state=1)

            # train on data
            clf.fit(X, y)

            # see predictions on the dataset
            clf.predict(test)#PAISA
            right_pred_1 = clf.score(test, test_lab)
            print("#### TEST 1 :\n\n")
            print(test[:10])

            print("#### TEST 1 :\n\n")
            print(f"Method: {solv}\tNb hidden layers: {str(hl)}\tAlpha: {str(a)}\n {right_pred_1}%\n\n")
            paisa_result.append(f"Method: {solv}\tNb hidden layers: {str(hl)}\tAlpha: {str(a)}\n {right_pred_1}%\n\n")

            clf.predict(test_2)#patterns
            print("#### TEST 2 :\n\n")
            print(test_2[:10])
            print("#### TEST 2 :\n\n")
            right_pred_2 = clf.score(test_2, test_temp_lab)
            template_result.append(f"Method: {solv}\tNb hidden layers: {str(hl)}\tAlpha: {str(a)}\n {right_pred_2}%\n\n")

            print(f"Method: {solv}\tNb hidden layers: {str(hl)}\tAlpha: {str(a)}\n {right_pred_2}%\n\n")

            print("\n###")

print("PAISA' TEST\n\n")
for scores in paisa_result:
    print(scores)

print("TEMPLATE TEST\n\n")
for scores in template_result:
    print(scores)
